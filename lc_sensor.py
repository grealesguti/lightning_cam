"""
lc_sensor.py -- thin wrapper around the AS3935 lightning sensor.

Design goals
------------
* Provide ONE class, AS3935Sensor, used by every script.
* Support SPI (as wired in the project plan) with I2C as an option.
* Never hard-crash if the hardware library isn't installed -- instead expose
  `.available` so test scripts can report clearly what's missing.
* Callback-on-IRQ model: register a Python callable that fires when the
  sensor pulls its IRQ line high.

Underlying library
-------------------
Uses `RPi_AS3935`-style access through SPI/I2C. Because several community
libraries exist with slightly different APIs, this wrapper talks to the chip
through a small internal driver over spidev/smbus so you are not locked to a
specific pip package. If you already use a library you trust, you can swap the
_Backend out; the public API (start/stop/read_event/last_registers) stays.

IMPORTANT (from the project plan): verify the *physical* GY-AS3935 board
pinout before wiring. Breakout boards relabel pins. This driver assumes the
SPI wiring in the README.
"""

import time
import threading

try:
    import spidev
    _HAVE_SPIDEV = True
except Exception:
    _HAVE_SPIDEV = False

# GPIO backend selection.
# Bookworm (kernel 6.x) dropped the sysfs GPIO interface that RPi.GPIO relies
# on, so RPi.GPIO's add_event_detect() fails with "Failed to add edge
# detection". lgpio talks to the modern gpiochip character device and is the
# correct library on current Pi OS. We prefer lgpio and fall back to RPi.GPIO
# on older systems.
_GPIO_BACKEND = None            # "lgpio" | "rpigpio" | None
try:
    import lgpio
    _GPIO_BACKEND = "lgpio"
except Exception:
    try:
        import RPi.GPIO as GPIO
        _GPIO_BACKEND = "rpigpio"
    except Exception:
        _GPIO_BACKEND = None

_HAVE_GPIO = _GPIO_BACKEND is not None


# AS3935 register / interrupt-reason constants
INT_NOISE = 0x01      # noise level too high
INT_DISTURBER = 0x04  # disturber detected (man-made noise)
INT_LIGHTNING = 0x08  # lightning detected

REG_INT = 0x03        # low nibble = interrupt reason
REG_DISTANCE = 0x07   # bits [5:0] = estimated distance (km)
REG_ENERGY_L = 0x04
REG_ENERGY_M = 0x05
REG_ENERGY_H = 0x06


class AS3935Sensor:
    def __init__(self, *, bus="spi", spi_bus=0, spi_dev=0, spi_hz=1_000_000,
                 i2c_addr=0x03, irq_gpio=17, indoor=True, logger=None):
        """
        bus       : "spi" or "i2c"
        spi_bus   : SPI bus number  (SPI0 -> 0)
        spi_dev   : chip-select     (CE0 -> 0)
        irq_gpio  : BCM pin number wired to the sensor IRQ
        indoor    : True sets the higher-gain indoor AFE preset; use False
                    outdoors (rooftop / field) to reduce false triggers.
        """
        self.bus_kind = bus
        self.spi_bus = spi_bus
        self.spi_dev = spi_dev
        self.spi_hz = spi_hz
        self.i2c_addr = i2c_addr
        self.irq_gpio = irq_gpio
        self.indoor = indoor
        self.log = logger

        self._spi = None
        self._i2c = None
        self._callback = None
        self._poll_thread = None
        self._stop = threading.Event()
        self._last_registers = {}

        # lgpio state (unused on the RPi.GPIO fallback)
        self._lg_handle = None
        self._lg_cb = None

        self.available = self._check_backend()

    # ------------------------------------------------------------------ #
    def _check_backend(self) -> bool:
        if self.bus_kind == "spi":
            if not _HAVE_SPIDEV:
                self._warn("spidev not installed -- `pip install spidev`")
                return False
        else:
            try:
                import smbus  # noqa
            except Exception:
                self._warn("smbus not installed -- `pip install smbus2`")
                return False
        if not _HAVE_GPIO:
            self._warn("No GPIO backend -- install lgpio (`pip install lgpio`, "
                       "recommended on Bookworm) or RPi.GPIO. IRQ unavailable.")
            return False
        return True

    def _warn(self, msg):
        if self.log:
            self.log.warning(msg)

    # ------------------------------------------------------------------ #
    # low-level register access
    # ------------------------------------------------------------------ #
    def _open(self):
        if self.bus_kind == "spi":
            self._spi = spidev.SpiDev()
            self._spi.open(self.spi_bus, self.spi_dev)
            self._spi.max_speed_hz = self.spi_hz
            self._spi.mode = 0b01          # AS3935 uses SPI mode 1
        else:
            import smbus
            self._i2c = smbus.SMBus(1)

    def _read_reg(self, reg) -> int:
        if self.bus_kind == "spi":
            # AS3935 read: set bit 6 (0x40) of the address byte
            resp = self._spi.xfer2([(reg & 0x3F) | 0x40, 0x00])
            return resp[1]
        else:
            return self._i2c.read_byte_data(self.i2c_addr, reg)

    def _write_reg(self, reg, val):
        if self.bus_kind == "spi":
            self._spi.xfer2([reg & 0x3F, val & 0xFF])
        else:
            self._i2c.write_byte_data(self.i2c_addr, reg, val & 0xFF)

    # ------------------------------------------------------------------ #
    # configuration
    # ------------------------------------------------------------------ #
    def configure(self):
        """Apply a sane default configuration and calibrate the RC oscillators."""
        # 0x3C = preset default, 0x3D = calibrate RCO
        self._write_reg(0x3C, 0x96)
        time.sleep(0.005)
        self._write_reg(0x3D, 0x96)
        time.sleep(0.005)

        # AFE gain: indoor (0x24>>1=0x12) vs outdoor (0x1C>>1=0x0E)
        afe = 0x12 if self.indoor else 0x0E
        reg0 = self._read_reg(0x00) & 0xC1
        self._write_reg(0x00, reg0 | (afe << 1))

        if self.log:
            self.log.info("AS3935 configured (%s preset).",
                          "indoor" if self.indoor else "outdoor")

    def set_noise_floor(self, level: int):
        """0..7 -- higher rejects more noise but also weaker strikes."""
        reg = self._read_reg(0x01) & 0x8F
        self._write_reg(0x01, reg | ((level & 0x07) << 4))

    # ------------------------------------------------------------------ #
    # event decoding
    # ------------------------------------------------------------------ #
    def read_event(self) -> dict:
        """
        Read and decode the interrupt cause. Call this after an IRQ.
        The AS3935 requires ~2 ms after IRQ before the INT reg is valid.
        """
        time.sleep(0.003)
        reason = self._read_reg(REG_INT) & 0x0F
        distance = self._read_reg(REG_DISTANCE) & 0x3F
        energy = ((self._read_reg(REG_ENERGY_H) & 0x1F) << 16 |
                  self._read_reg(REG_ENERGY_M) << 8 |
                  self._read_reg(REG_ENERGY_L))

        kind = {
            INT_NOISE: "noise",
            INT_DISTURBER: "disturber",
            INT_LIGHTNING: "lightning",
        }.get(reason, f"unknown(0x{reason:02X})")

        ev = {
            "time": time.time(),
            "reason_code": reason,
            "kind": kind,
            "distance_km": None if distance == 0x3F else distance,  # 0x3F = out of range
            "energy": energy,
        }
        self._last_registers = ev
        return ev

    @property
    def last_registers(self) -> dict:
        return dict(self._last_registers)

    # ------------------------------------------------------------------ #
    # IRQ handling
    # ------------------------------------------------------------------ #
    def start(self, callback):
        """
        Open the bus, configure the chip, and begin listening for IRQs.
        `callback(event_dict)` is invoked on each interrupt.

        Uses lgpio (gpiochip character device) on Bookworm, falling back to
        RPi.GPIO on older systems.
        """
        if not self.available:
            raise RuntimeError("Sensor backend unavailable -- see warnings above.")

        self._callback = callback
        self._open()
        self.configure()

        if _GPIO_BACKEND == "lgpio":
            self._start_lgpio()
        elif _GPIO_BACKEND == "rpigpio":
            self._start_rpigpio()
        else:
            raise RuntimeError("No usable GPIO backend.")

        if self.log:
            self.log.info("AS3935 listening on GPIO%d (IRQ, backend=%s).",
                          self.irq_gpio, _GPIO_BACKEND)

    # --- lgpio backend (Bookworm / current Pi OS) -------------------------- #
    def _start_lgpio(self):
        # gpiochip0 is the main header bank on Pi 3/4; Pi 5 uses gpiochip4 for
        # the 40-pin header. Try 0 first, then 4.
        handle = None
        last_err = None
        for chip in (0, 4):
            try:
                handle = lgpio.gpiochip_open(chip)
                break
            except Exception as e:
                last_err = e
        if handle is None:
            raise RuntimeError(f"Could not open a gpiochip: {last_err}")
        self._lg_handle = handle

        # claim IRQ line for rising-edge alerts, with a pull-down and small
        # debounce so a single strike doesn't fire the callback repeatedly.
        lgpio.gpio_claim_alert(self._lg_handle, self.irq_gpio,
                               lgpio.RISING_EDGE, lgpio.SET_PULL_DOWN)
        try:
            lgpio.gpio_set_debounce_micros(self._lg_handle, self.irq_gpio, 5000)
        except Exception:
            pass  # older lgpio may lack this; harmless

        # lgpio delivers alerts to a callback with this signature:
        #   cb(chip, gpio, level, tick)
        self._lg_cb = lgpio.callback(self._lg_handle, self.irq_gpio,
                                     lgpio.RISING_EDGE, self._on_irq_lgpio)

    def _on_irq_lgpio(self, _chip, _gpio, _level, _tick):
        self._dispatch_irq()

    # --- RPi.GPIO backend (older systems) --------------------------------- #
    def _start_rpigpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.irq_gpio, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.add_event_detect(self.irq_gpio, GPIO.RISING,
                              callback=self._on_irq_rpigpio, bouncetime=5)

    def _on_irq_rpigpio(self, _channel):
        self._dispatch_irq()

    # --- shared dispatch --------------------------------------------------- #
    def _dispatch_irq(self):
        try:
            ev = self.read_event()
            if self._callback:
                self._callback(ev)
        except Exception as e:
            if self.log:
                self.log.error("IRQ handler error: %s", e)

    def stop(self):
        self._stop.set()
        if _GPIO_BACKEND == "lgpio":
            try:
                if self._lg_cb is not None:
                    self._lg_cb.cancel()
            except Exception:
                pass
            try:
                if self._lg_handle is not None:
                    lgpio.gpio_free(self._lg_handle, self.irq_gpio)
                    lgpio.gpiochip_close(self._lg_handle)
            except Exception:
                pass
        elif _GPIO_BACKEND == "rpigpio":
            try:
                GPIO.remove_event_detect(self.irq_gpio)
                GPIO.cleanup(self.irq_gpio)
            except Exception:
                pass
        if self._spi:
            self._spi.close()
        if self.log:
            self.log.info("AS3935 stopped.")
