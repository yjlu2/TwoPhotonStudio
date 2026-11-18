"""Run analysis outside the GUI process so native calls and child models can stop."""
import contextlib
import ctypes
import multiprocessing as mp
from pathlib import Path
import re
import threading
import traceback
from ctypes import wintypes

from PyQt5 import QtCore
from error_catalog import error_summary, open_diagnostic_log, write_diagnostic_log


class StepLog:
    def __init__(self, pipe):
        self.pipe=pipe
        self.lock=threading.Lock()

    def send(self, kind, value):
        with self.lock:
            self.pipe.send((kind,value))

    def write(self, message):
        if message.strip(): self.send('log',message.rstrip())

    __call__=write

    def temporary(self, path):
        self.send('temporary',str(path))

    def temporary_tree(self, path, token):
        self.send('temporary_tree',(str(path),token))

    def flush(self): pass


def run_task(task, pipe, start):
    # No work or child launch until the parent assigns this process to a Windows job.
    start.wait()
    log=StepLog(pipe)
    with pipe, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            log.send('done',task(log=log))
        except BaseException:
            log.send('error',traceback.format_exc())


class ProcessJob:
    def __init__(self, pid):
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        signatures={
            'CreateJobObjectW':([ctypes.c_void_p,wintypes.LPCWSTR],wintypes.HANDLE),
            'OpenProcess':([wintypes.DWORD,wintypes.BOOL,wintypes.DWORD],wintypes.HANDLE),
            'AssignProcessToJobObject':([wintypes.HANDLE,wintypes.HANDLE],wintypes.BOOL),
            'TerminateJobObject':([wintypes.HANDLE,wintypes.UINT],wintypes.BOOL),
            'CloseHandle':([wintypes.HANDLE],wintypes.BOOL),
        }
        for name,(args,result) in signatures.items():
            fn=getattr(kernel,name); fn.argtypes=args; fn.restype=result
        self.kernel=kernel
        self.handle=kernel.CreateJobObjectW(None,None)
        if not self.handle: raise ctypes.WinError(ctypes.get_last_error())
        process=kernel.OpenProcess(0x0101,False,pid)
        try:
            if not process or not kernel.AssignProcessToJobObject(self.handle,process):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.close()
            raise
        finally:
            if process: kernel.CloseHandle(process)

    def close(self):
        if self.handle:
            self.kernel.TerminateJobObject(self.handle,1)
            self.kernel.CloseHandle(self.handle)
            self.handle=None




class Worker(QtCore.QThread):
    log=QtCore.pyqtSignal(str)
    done=QtCore.pyqtSignal(object)
    failed=QtCore.pyqtSignal(str,str)
    stopped=QtCore.pyqtSignal()

    def __init__(self,task,log_path):
        super().__init__(); self.task=task; self.log_path=log_path
        self.cancel=threading.Event()

    def run(self):
        process=job=reader=writer=transcript=None
        temporary=[]; temporary_trees=[]; detail=None; finished=False; result=None
        def record(message):
            transcript.write(message+'\n'); transcript.flush()
        try:
            self.log_path,transcript=open_diagnostic_log(self.log_path)
            ctx=mp.get_context('spawn')
            reader,writer=ctx.Pipe(duplex=False); start=ctx.Event()
            process=ctx.Process(target=run_task,args=(self.task,writer,start))
            process.start(); writer.close()
            job=ProcessJob(process.pid)
            start.set()
            while not self.cancel.is_set():
                if reader.poll(.1):
                    try: kind,value=reader.recv()
                    except EOFError: break
                    if kind=='log': record(value); self.log.emit(value)
                    elif kind=='temporary': temporary.append(Path(value))
                    elif kind=='temporary_tree': temporary_trees.append(value)
                    elif kind=='error': detail=value; break
                    elif kind=='done': finished=True; result=value; break
                elif not process.is_alive(): break
            if self.cancel.is_set(): record('Stopped by user.')
            elif not finished and detail is None:
                detail=f'The worker exited unexpectedly (exit code {process.exitcode}).'
        except Exception:
            detail=traceback.format_exc()
        finally:
            if process is not None and process.pid is not None:
                if finished: process.join(timeout=1)
                if job is not None: job.close()
                elif process.is_alive(): process.terminate()
                process.join()
            if reader is not None: reader.close()
            if writer is not None: writer.close()
            for path in temporary:
                try: path.unlink(missing_ok=True)
                except OSError: pass
            for path,token in temporary_trees:
                try:
                    from output_location import cleanup_owned_copy
                    cleanup_owned_copy(path,token)
                except Exception:
                    self.log.emit('Incomplete output-copy folder retained for inspection: '+str(path))
            if transcript is not None:
                try:
                    if detail: record(detail)
                except OSError:
                    if detail:
                        try: self.log_path=write_diagnostic_log(detail)
                        except OSError: transcript=None
                finally:
                    if transcript is not None: transcript.close()
        if self.cancel.is_set(): self.stopped.emit()
        elif detail: self.failed.emit(error_summary(detail),str(self.log_path) if transcript is not None else '')
        elif finished: self.done.emit(result)
