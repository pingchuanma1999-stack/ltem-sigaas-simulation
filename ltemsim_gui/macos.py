"""macOS integration without PyObjC: give the running Python process the app's name in the menu bar."""
from __future__ import annotations

import ctypes
import ctypes.util
import sys


def set_app_name(name: str):
    """Set CFBundleName of the main bundle (must run before the QApplication is created)."""
    if sys.platform != "darwin":
        return
    try:
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        send = objc.objc_msgSend

        def msg(obj, sel, *args, argtypes=()):
            send.restype = ctypes.c_void_p
            send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *argtypes]
            return send(obj, objc.sel_registerName(sel), *args)

        def nsstr(s: str):
            return msg(objc.objc_getClass(b"NSString"), b"stringWithUTF8String:", s.encode("utf-8"),
                       argtypes=(ctypes.c_char_p,))

        bundle = msg(objc.objc_getClass(b"NSBundle"), b"mainBundle")
        for getter in (b"localizedInfoDictionary", b"infoDictionary"):
            info = msg(bundle, getter)
            if info:
                msg(info, b"setObject:forKey:", nsstr(name), nsstr("CFBundleName"),
                    argtypes=(ctypes.c_void_p, ctypes.c_void_p))
    except Exception:
        pass
