import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSFloatingWindowLevel,
    NSMakeRect,
    NSPanel,
    NSScreen,
    NSView,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from PyObjCTools import AppHelper

WINDOW_WIDTH = 220
WINDOW_HEIGHT = 60
BOTTOM_MARGIN = 80
MAX_BARS = 40


def _normalize_level(dbfs: float, floor_dbfs: float = -60.0) -> float:
    if dbfs <= floor_dbfs:
        return 0.0
    if dbfs >= 0.0:
        return 1.0
    return (dbfs - floor_dbfs) / (0.0 - floor_dbfs)


class WaveformView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(WaveformView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.levels = []
        return self

    def drawRect_(self, rect):
        bounds = self.bounds()

        background = NSColor.colorWithCalibratedWhite_alpha_(0.1, 0.85)
        background.setFill()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(bounds, 12, 12).fill()

        if not self.levels:
            return

        bar_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.3, 0.85, 0.5, 1.0)
        bar_color.setFill()

        width = bounds.size.width
        height = bounds.size.height
        bar_width = width / MAX_BARS
        for i, level in enumerate(self.levels):
            normalized = _normalize_level(level)
            bar_height = max(2, normalized * (height - 12))
            x = i * bar_width
            y = (height - bar_height) / 2
            bar_rect = NSMakeRect(x + 1, y, max(1, bar_width - 2), bar_height)
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(bar_rect, 2, 2).fill()


class WaveformWindowController:
    def __init__(self):
        self.panel = None
        self.view = None

    def _build(self):
        screen_frame = NSScreen.mainScreen().frame()
        x = (screen_frame.size.width - WINDOW_WIDTH) / 2
        rect = NSMakeRect(x, BOTTOM_MARGIN, WINDOW_WIDTH, WINDOW_HEIGHT)
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, NSBackingStoreBuffered, False
        )
        panel.setLevel_(NSFloatingWindowLevel)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setIgnoresMouseEvents_(True)

        view = WaveformView.alloc().initWithFrame_(NSMakeRect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT))
        panel.setContentView_(view)

        self.panel = panel
        self.view = view

    def show(self):
        def _show():
            if self.panel is None:
                self._build()
            self.view.levels = []
            self.panel.orderFrontRegardless()

        AppHelper.callAfter(_show)

    def hide(self):
        def _hide():
            if self.panel is not None:
                self.panel.orderOut_(None)

        AppHelper.callAfter(_hide)

    def push_level(self, level):
        def _update():
            if self.view is None:
                return
            self.view.levels.append(level)
            if len(self.view.levels) > MAX_BARS:
                self.view.levels = self.view.levels[-MAX_BARS:]
            self.view.setNeedsDisplay_(True)

        AppHelper.callAfter(_update)
