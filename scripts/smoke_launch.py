"""Exercise the installed command entry point, then automatically close the GUI."""
import os
import tempfile
from pathlib import Path
from functools import partial
os.environ['QT_QPA_PLATFORM']='offscreen'
from twophoton_studio.cli import configure_dlls, main
configure_dlls()
from PyQt5 import QtWidgets, QtCore

original_exec=QtWidgets.QApplication.exec_

def checked_exec(self):
    windows=[w for w in self.topLevelWidgets() if isinstance(w,QtWidgets.QMainWindow)]
    assert any('TwoPhoton Studio' in w.windowTitle() for w in windows)
    print('Installed twophoton-studio entry point opened the GUI.',flush=True)
    import numpy as np
    import tifffile
    import backend
    from step_worker import Worker
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.local') as folder:
        folder=Path(folder)
        tif=folder/'test.tif'
        tifffile.imwrite(tif,np.ones((2,16,16),dtype='uint16'),photometric='minisblack')
        outcome={}
        worker=Worker(partial(backend.import_tiffs,[str(tif)],folder/'session'),folder/'worker.log')
        def done(result):
            outcome['result']=result
            QtCore.QTimer.singleShot(100,self.quit)
        def failed(message,path):
            outcome['error']=message+' | '+path
            self.quit()
        worker.done.connect(done); worker.failed.connect(failed)
        worker.log.connect(lambda line:print(line,flush=True))
        timeout=QtCore.QTimer(); timeout.setSingleShot(True)
        timeout.timeout.connect(self.quit); timeout.start(120000)
        worker.start()
        code=original_exec()
        worker.cancel.set(); worker.wait()
        assert 'result' in outcome, outcome.get('error','Worker timed out')
        print('Spawned GUI worker imported a synthetic TIFF successfully.',flush=True)
        return code

QtWidgets.QApplication.exec_=checked_exec
main()
