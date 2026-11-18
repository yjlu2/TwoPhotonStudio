import os
import runpy
import sys
from pathlib import Path

_DLL_HANDLES=[]

def configure_dlls():
    prefix=Path(sys.prefix)
    folders=[prefix,prefix/'Library/bin',prefix/'DLLs',prefix/'Scripts']
    os.environ['PATH']=os.pathsep.join(str(p) for p in folders)+os.pathsep+os.environ.get('PATH','')
    if hasattr(os,'add_dll_directory'):
        for path in folders:
            if path.is_dir(): _DLL_HANDLES.append(os.add_dll_directory(str(path)))

def main():
    if os.name!='nt':
        raise SystemExit('This release supports Windows x64. Linux/macOS are not yet supported.')
    os.environ['PYTHONNOUSERSITE']='1'
    os.environ['PYTHONUTF8']='1'
    os.environ['QT_API']='pyqt5'
    configure_dlls()
    here=Path(__file__).resolve().parent
    sys.path.insert(0,str(here))
    import multiprocessing
    multiprocessing.freeze_support()
    from runtime_config import installation
    if not installation():
        raise SystemExit('Run python install.py --device cuda (or cpu) from the repository first.')
    runpy.run_path(str(here/'studio.py'),run_name='__main__')

if __name__ == '__main__':
    main()

