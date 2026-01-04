import win32gui

def enumHandler(hwnd, ctx):
    if win32gui.IsWindowVisible(hwnd):
        title = win32gui.GetWindowText(hwnd)
        if title.strip():
            print(title)

win32gui.EnumWindows(enumHandler, None)
