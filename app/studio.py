"""Independent Suite2p-based desktop workflow; no GetROI code dependency."""
from functools import partial
import copy
import json
import os
import sys
import traceback
import uuid
import time
import shutil
import re
from pathlib import Path
import numpy as np
import tifffile
from scipy.ndimage import binary_erosion
from PyQt5 import QtCore, QtGui, QtWidgets as Q
import pyqtgraph as pg
import backend as B
from display_contrast import ContrastDialog
from step_worker import Worker, error_summary
from error_catalog import write_diagnostic_log
from copyable_paths import PathTable, PathList

pg.setConfigOptions(imageAxisOrder='row-major', antialias=False)

class Settings(Q.QDialog):
    def __init__(self, cfg, parent):
        super().__init__(parent); self.setWindowTitle('Algorithm and environment settings'); self.resize(820,760)
        root=Q.QVBoxLayout(self); tabs=Q.QTabWidget(); root.addWidget(tabs); self.fields={}
        specs=[('Analysis',[
            ('fs','Frame rate / plane (Hz)',.01,10000.),('tau','Indicator tau (s)',.01,100.),
            ('diameter','Cell diameter (px; Cellpose)',2,300),
            ('threshold_scaling','Detection threshold (lower = more ROIs)',.05,10.),
            ('block_size','Nonrigid block size (px; even)',16,512),
            ('maxregshift','Max rigid shift / image size',.001,.4),
            ('maxregshiftNR','Max nonrigid shift (px)',1,50),
            ('inner_neuropil_radius','Inner neuropil radius (px)',0,100),
            ('min_neuropil_pixels','Minimum neuropil pixels',10,10000),
            ('neucoeff','Neuropil coefficient',0.,1.),
            ('win_baseline','Baseline window (s)',1.,1000.),
            ('sig_baseline','Baseline smoothing (frames)',0.,100.),
            ('trial_baseline_seconds','Trial z-score baseline before onset (s)',.05,60.),
            ('trial_post_seconds','Trial heatmap time after onset (s)',.1,600.),
            ('merge_iou','Duplicate candidate IoU threshold',.1,.95)]),
            ('Models and environments',[(k,l,None,None) for k,l in [
            ('matlab','MATLAB executable'),('reg_code','reg_stacks / dftregistration directory'),
            ('suite2p_gpu_python','Suite2p GPU Python (1.x)'),
            ('support_python','SUPPORT Python'),('support_root','SUPPORT repository'),
            ('support_checkpoint','SUPPORT checkpoint (.pth)'),('cellpose_python','Cellpose Python'),
            ('cellpose_model','Cellpose checkpoint (local)'),('profile_python','PROFILE Python'),
            ('profile_root','PROFILE repository'),('profile_config','PROFILE matching config (.json)'),
            ('profile_checkpoint','PROFILE checkpoint (optional override)')]])]
        for title,rows in specs:
            page=Q.QWidget(); form=Q.QFormLayout(page); tabs.addTab(page,title)
            for key,label,lo,hi in rows:
                if lo is None:
                    field=Q.QLineEdit(str(cfg[key])); holder=Q.QWidget(); line=Q.QHBoxLayout(holder); line.setContentsMargins(0,0,0,0)
                    line.addWidget(field); browse=Q.QPushButton('…'); browse.setFixedWidth(30); line.addWidget(browse)
                    browse.clicked.connect(lambda _,f=field,k=key:self.browse(f,k)); widget=holder
                else:
                    field=Q.QSpinBox() if isinstance(lo,int) and isinstance(hi,int) else Q.QDoubleSpinBox()
                    if isinstance(field,Q.QDoubleSpinBox): field.setDecimals(3)
                    field.setRange(lo,hi); field.setValue(cfg[key]); widget=field
                self.fields[key]=field; form.addRow(label,widget)
            if title=='Analysis':
                shape=Q.QComboBox()
                shape.addItem('Circular expansion',True)
                shape.addItem('Rectangular expansion',False)
                shape.setCurrentIndex(0 if cfg.get('circular_neuropil',True) else 1)
                shape.setToolTip('Expand along the ROI outline or a rectangular bounding area. Changing this setting requires recomputing traces.')
                self.fields['circular_neuropil']=shape
                form.addRow('Neuropil mask shape',shape)
        self.device=Q.QComboBox(); self.device.addItems(['cuda','cpu']); self.device.setCurrentText(cfg['device'])
        self.bs=Q.QSpinBox(); self.bs.setRange(1,9); self.bs.setSingleStep(2); self.bs.setValue(cfg['support_bs'])
        form.addRow('Inference device',self.device); form.addRow('SUPPORT blind spot (match checkpoint)',self.bs)
        sync_page=Q.QWidget(); sync_form=Q.QFormLayout(sync_page); tabs.addTab(sync_page,'Synchronization MAT')
        variable=Q.QLineEdit(str(cfg.get('sync_variable','data2026'))); self.fields['sync_variable']=variable
        sync_form.addRow('Matrix variable name',variable)
        for key,label,maximum in [('sync_frame_column','Frame TTL column (1-based)',256),
                                  ('sync_stim_column','Stimulus TTL column (1-based)',256)]:
            field=Q.QSpinBox(); field.setRange(1,maximum); field.setValue(int(cfg.get(key,1)))
            self.fields[key]=field; sync_form.addRow(label,field)
        sample_rate=Q.QDoubleSpinBox(); sample_rate.setDecimals(3); sample_rate.setRange(1.,1000000.)
        sample_rate.setValue(float(cfg.get('sync_sample_rate',1000.))); self.fields['sync_sample_rate']=sample_rate
        sync_form.addRow('NI sample rate (Hz)',sample_rate)
        alignment=Q.QComboBox(); alignment.addItem('Strict: one frame TTL per TIFF frame','strict')
        alignment.addItem('First TIFF frame = first TTL rise; remaining frames use nominal Hz','first_edge')
        alignment.setCurrentIndex(max(0,alignment.findData(cfg.get('sync_alignment','strict'))))
        self.fields['sync_alignment']=alignment; sync_form.addRow('Frame alignment',alignment)
        sync_note=Q.QLabel('TTL high intervals are detected automatically. Both rising and falling edges are retained.\nDefaults: data2026, frame column 1, stimulus column 2. For the supplied data2023 file use columns 1 and 4.')
        sync_note.setWordWrap(True); sync_form.addRow(sync_note)
        helptext=Q.QLabel('Select a PROFILE config and checkpoint matching your imaging scale.\nSUPPORT is used for detection only. Traces use the registered movie without denoising.')
        helptext.setWordWrap(True); root.addWidget(helptext)
        buttons=Q.QDialogButtonBox(Q.QDialogButtonBox.Ok|Q.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); root.addWidget(buttons)
    def browse(self,field,key):
        path=Q.QFileDialog.getExistingDirectory(self,'Select directory') if key in ('reg_code','support_root','profile_root') else Q.QFileDialog.getOpenFileName(self,'Select file')[0]
        if path: field.setText(path)
    def values(self):
        result={k:(f.text() if isinstance(f,Q.QLineEdit) else f.currentData() if isinstance(f,Q.QComboBox) else f.value()) for k,f in self.fields.items()}
        result.update(device=self.device.currentText(),support_bs=self.bs.value()); return result

class ImportStacksDialog(Q.QDialog):
    """Collect same-FOV stacks across any number of directories before import."""
    def __init__(self, paths, parent=None):
        super().__init__(parent); self.setWindowTitle('Import TIFF stacks - same FOV'); self.resize(900,460)
        root=Q.QVBoxLayout(self)
        note=Q.QLabel('Add TIFF stacks from one or more folders. The displayed order is the analysis and frame order.\nOriginal TIFFs will be linked, not copied. Use single-plane, single-channel files from one FOV.')
        note.setWordWrap(True); root.addWidget(note)
        self.files=PathList(); self.files.setSelectionMode(Q.QAbstractItemView.ExtendedSelection); root.addWidget(self.files,1)
        actions=Q.QHBoxLayout(); root.addLayout(actions)
        add=Q.QPushButton('Add TIFF files...'); add.clicked.connect(self.choose_more); actions.addWidget(add)
        remove=Q.QPushButton('Remove selected'); remove.clicked.connect(self.remove_selected); actions.addWidget(remove)
        up=Q.QPushButton('Move up'); up.clicked.connect(lambda:self.move(-1)); actions.addWidget(up)
        down=Q.QPushButton('Move down'); down.clicked.connect(lambda:self.move(1)); actions.addWidget(down)
        actions.addStretch()
        buttons=Q.QDialogButtonBox(Q.QDialogButtonBox.Ok|Q.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); root.addWidget(buttons)
        self.add_paths(paths)
    def paths(self): return [self.files.item(i).text() for i in range(self.files.count())]
    def add_paths(self, paths):
        known={os.path.normcase(p) for p in self.paths()}
        for path in sorted((str(Path(p).resolve()) for p in paths),key=B.natural):
            if os.path.normcase(path) not in known:
                self.files.addItem(path); known.add(os.path.normcase(path))
    def choose_more(self):
        paths,_=Q.QFileDialog.getOpenFileNames(self,'Add TIFF stacks from another folder','','TIFF (*.tif *.tiff)')
        if paths: self.add_paths(paths)
    def remove_selected(self):
        for item in sorted(self.files.selectedItems(),key=lambda x:self.files.row(x),reverse=True): self.files.takeItem(self.files.row(item))
    def move(self, step):
        rows=sorted({self.files.row(i) for i in self.files.selectedItems()},reverse=step>0)
        for row in rows:
            target=row+step
            if 0<=target<self.files.count():
                item=self.files.takeItem(row); self.files.insertItem(target,item); item.setSelected(True)
    def accept(self):
        if not self.files.count():
            Q.QMessageBox.warning(self,'No TIFF files','Add at least one TIFF stack.'); return
        super().accept()

class FitImageLabel(Q.QLabel):
    def __init__(self, image_path, parent=None):
        super().__init__(parent); self.source=QtGui.QPixmap(str(image_path))
        self.setAlignment(QtCore.Qt.AlignCenter); self.setMinimumSize(320,240)
        self.setSizePolicy(Q.QSizePolicy.Ignored,Q.QSizePolicy.Ignored); self.update_image()
    def update_image(self):
        if not self.source.isNull() and self.width()>0 and self.height()>0:
            self.setPixmap(self.source.scaled(self.size(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
    def resizeEvent(self,event):
        super().resizeEvent(event); self.update_image()

class HeatmapWindow(Q.QDialog):
    def __init__(self, image_path, fig_path, parent=None):
        super().__init__(parent); self.setWindowTitle('Trial response heatmap'); self.resize(1150,850)
        self.fig_path=Path(fig_path)
        layout=Q.QVBoxLayout(self); self.image_label=FitImageLabel(image_path,self); layout.addWidget(self.image_label,1)
        row=Q.QHBoxLayout(); layout.addLayout(row); row.addStretch()
        export=Q.QPushButton('Export MATLAB FIG...'); export.clicked.connect(self.export_fig); row.addWidget(export)
        close=Q.QPushButton('Close'); close.clicked.connect(self.accept); row.addWidget(close)
    def export_fig(self):
        path,_=Q.QFileDialog.getSaveFileName(self,'Export editable MATLAB figure',self.fig_path.name,'MATLAB figure (*.fig)')
        if not path: return
        destination=Path(path).with_suffix('.fig')
        if destination.exists() and destination.resolve()!=self.fig_path.resolve():
            if Q.QMessageBox.question(self,'Replace file',f'Replace {destination}?')!=Q.QMessageBox.Yes: return
        if destination.resolve()!=self.fig_path.resolve(): shutil.copy2(self.fig_path,destination)

class Studio(Q.QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle('TwoPhoton Studio 1.3.4 | Suite2p workflow'); self.resize(1500,960)
        self.display_levels={}; self.contrast_dialog=None
        self.cfg=B.defaults(); self.session=dict(root='',ops_file='',stat=[],result='',frame_times=None)
        self.setAcceptDrops(True)
        self.worker=None; self.dirty=True; self.selected=-1; self.traces={}; self.adding=False; self.polygon=[]; self.pending=0
        self.roi_revision=0; self.mean=None; self.movie=None; self.valid_area=None; self.valid_bounds=None; self.controls=[]
        self.undo_rois=None
        self.edge_cache={}
        self.total_seconds=0.; self.analysis_windows=[]
        main=Q.QWidget(); self.setCentralWidget(main); layout=Q.QHBoxLayout(main); layout.setContentsMargins(6,6,6,6)
        split=Q.QSplitter(QtCore.Qt.Horizontal); layout.addWidget(split)
        left=Q.QWidget(); left.setMinimumWidth(330); left.setMaximumWidth(460); ll=Q.QVBoxLayout(left); ll.setContentsMargins(0,0,4,0)
        self.left_tabs=Q.QTabWidget(); ll.addWidget(self.left_tabs,1)
        workflow=Q.QWidget(); wl=Q.QVBoxLayout(workflow); wl.setContentsMargins(6,6,6,6); self.left_tabs.addTab(workflow,'Workflow')
        row=Q.QHBoxLayout(); wl.addLayout(row)
        self.button(row,'Import TIFF stacks',self.import_files); self.button(row,'Import TIFF folder',self.import_folder)
        row=Q.QHBoxLayout(); wl.addLayout(row)
        self.button(row,'Open session',self.load_session); self.button(row,'Save session',self.save_session)
        row=Q.QHBoxLayout(); wl.addLayout(row)
        self.param_label=Q.QLabel(f'{self.cfg["fs"]:g} Hz  |  tau {self.cfg["tau"]:g} s'); row.addWidget(self.param_label,1)

        registration_box=Q.QGroupBox('Registration and denoising'); options=Q.QFormLayout(registration_box); wl.addWidget(registration_box)
        self.s2p_backend=Q.QComboBox(); self.s2p_backend.addItems(['Suite2p CPU','Suite2p GPU (reg / extract)'])
        self.s2p_backend.setCurrentIndex(1 if self.cfg.get('suite2p_backend')=='gpu' else 0)
        options.addRow('Backend',self.s2p_backend); self.controls.append(self.s2p_backend); self.s2p_backend.currentIndexChanged.connect(self.prep_changed)
        self.reg=Q.QComboBox(); self.reg.addItems(['Suite2p nonrigid','MATLAB reg_stacks (DFT)','Already registered'])
        self.reg.currentIndexChanged.connect(self.reg_changed); options.addRow('Method',self.reg); self.controls.append(self.reg)
        self.support=Q.QCheckBox('SUPPORT after registration'); self.support.toggled.connect(self.prep_changed); options.addRow(self.support); self.controls.append(self.support)
        self.continuous=Q.QCheckBox('TIFFs are continuous segments'); self.continuous.setChecked(True)
        self.continuous.toggled.connect(self.prep_changed); options.addRow(self.continuous); self.controls.append(self.continuous)
        self.registered_tiff=Q.QComboBox()
        self.registered_tiff.addItems(['Separate TIFFs','Combined TIFF','Do not save'])
        self.registered_tiff.setToolTip('Separate: each original TIFF folder gets <source-name>_Registered.tif; existing files get a timestamp suffix. Combined: one Registered.tif under the analysis preparation folder. Input frame order is preserved.')
        self.controls.append(self.registered_tiff); options.addRow('Registered TIFF',self.registered_tiff)
        self.button(wl,'1  Register / Denoise',self.prepare)
        detection_box=Q.QGroupBox('ROI detection'); dl=Q.QVBoxLayout(detection_box); wl.addWidget(detection_box)
        self.detector=Q.QComboBox(); self.detector.addItems(['Suite2p','Cellpose','PROFILE']); dl.addWidget(self.detector); self.controls.append(self.detector)
        self.button(dl,'2  Detect / Merge ROIs',self.detect)
        self.button(dl,'Choose PROFILE JSON...',self.choose_profile)
        self.profile_label=Q.QLabel('PROFILE: no config selected'); self.profile_label.setWordWrap(True); self.profile_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse); dl.addWidget(self.profile_label)
        self.replace_detector=Q.QCheckBox('Replace previous ROIs from this detector'); dl.addWidget(self.replace_detector); self.controls.append(self.replace_detector)
        self.button(wl,'3  Compute traces',self.calculate)
        tools=Q.QMenu(self)
        self.tools_button=Q.QToolButton()
        self.tools_button.setText('Tools')
        self.tools_button.setMenu(tools)
        self.tools_button.setPopupMode(Q.QToolButton.InstantPopup)
        self.tools_button.setMinimumHeight(28)
        header=Q.QWidget(); header_layout=Q.QHBoxLayout(header)
        header_layout.setContentsMargins(0,0,0,0); header_layout.setSpacing(2)
        header_layout.addWidget(self.tools_button)
        self.settings_button=Q.QToolButton(); self.settings_button.setText('Settings')
        self.settings_button.setMinimumHeight(28)
        self.settings_button.clicked.connect(lambda:self.guard(self.settings))
        self.controls.append(self.settings_button); header_layout.addWidget(self.settings_button)
        self.left_tabs.setCornerWidget(header,QtCore.Qt.TopRightCorner)
        for title,fn in [('Import label mask...',self.import_label_mask),
                         ('Import manual neuropil / re-extract...',self.import_manual_neuropil),
                         ('Common errors...',self.show_error_reference)]:
            action=tools.addAction(title)
            action.triggered.connect(lambda checked=False,fn=fn:self.guard(fn))
            self.controls.append(action)
        self.button(wl,'dF/F trace heatmap',self.show_trace_heatmap)
        row=Q.QHBoxLayout(); wl.addLayout(row); self.button(row,'4  Export Fall.mat',self.export); self.button(row,'Import sync MAT',self.import_times)
        self.trial_analysis_button=self.button(wl,'5  Trial response heatmap',self.analyze_trials)
        wl.addStretch()

        self.roi_widget=Q.QWidget(); sl=Q.QVBoxLayout(self.roi_widget); sl.setContentsMargins(6,6,6,6); self.left_tabs.addTab(self.roi_widget,'ROIs')
        self.table=Q.QTableWidget(0,4); self.table.setHorizontalHeaderLabels(['ID','Source','Class','P(cell)'])
        self.table.setSelectionBehavior(Q.QAbstractItemView.SelectRows); self.table.setSelectionMode(Q.QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(Q.QAbstractItemView.NoEditTriggers); self.table.cellClicked.connect(self.select_row); sl.addWidget(self.table,1)
        row=Q.QHBoxLayout(); sl.addLayout(row); self.button(row,'Mark cell',lambda:self.label(True)); self.button(row,'Mark not cell',lambda:self.label(False))
        row=Q.QHBoxLayout(); sl.addLayout(row); self.button(row,'Delete selected ROIs',self.delete); self.button(row,'Undo delete',self.undo_delete)
        self.button(sl,'Start manual Add',self.start_add)
        row=Q.QHBoxLayout(); sl.addLayout(row)
        self.shape=Q.QComboBox(); self.shape.addItems(['Circle','Polygon (right-click, auto-close)']); row.addWidget(self.shape); self.controls.append(self.shape)
        self.shape.setToolTip('Polygon: hold the right mouse button and trace the cell boundary; release to close and add the ROI.')
        row.addWidget(Q.QLabel('Radius')); self.radius=Q.QDoubleSpinBox(); self.radius.setRange(1,100); self.radius.setValue(6); row.addWidget(self.radius); self.controls.append(self.radius)
        self.button(sl,'Finish Add + Compute',self.finish_add)
        hint=Q.QLabel('Ctrl / Shift selects multiple rows. Delete removes selected ROIs; Undo delete restores them. Add collects masks; Finish Add computes traces.'); hint.setWordWrap(True); sl.addWidget(hint)

        self.source_widget=Q.QWidget(); source_layout=Q.QVBoxLayout(self.source_widget); source_layout.setContentsMargins(6,6,6,6)
        self.left_tabs.addTab(self.source_widget,'Sources')
        self.source_summary=Q.QLabel('No TIFF files imported'); self.source_summary.setWordWrap(True)
        self.source_summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse|QtCore.Qt.TextSelectableByKeyboard)
        self.source_summary.setFocusPolicy(QtCore.Qt.StrongFocus); source_layout.addWidget(self.source_summary)
        self.button(source_layout,'Add TIFF files...',self.add_source_tiffs)
        self.button(source_layout,'Change output folder...',self.change_output_folder)
        self.source_table=PathTable(); self.source_table.setHorizontalHeaderLabels(['Order','Frames','Full source path'])
        self.source_table.setEditTriggers(Q.QAbstractItemView.NoEditTriggers); self.source_table.horizontalHeader().setStretchLastSection(True); source_layout.addWidget(self.source_table,1)

        logbar=Q.QHBoxLayout(); ll.addLayout(logbar)
        logbar.addWidget(Q.QLabel('Processing log'),1)
        self.stop_button=Q.QPushButton('Stop'); self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_step); logbar.addWidget(self.stop_button)
        self.log=Q.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(110); self.log.setMaximumHeight(180); ll.addWidget(self.log)
        self.device_label=Q.QLabel('Suite2p: GPU requested | Neural models: CUDA requested'); self.device_label.setWordWrap(True); ll.addWidget(self.device_label)
        self.elapsed_label=Q.QLabel('Ready'); ll.addWidget(self.elapsed_label)
        self.progress=Q.QProgressBar(); self.progress.setRange(0,1); self.progress.setValue(0); self.progress.setMaximumHeight(18); ll.addWidget(self.progress)
        split.addWidget(left)

        view=Q.QWidget(); vl=Q.QVBoxLayout(view); vl.setContentsMargins(4,0,0,0); split.addWidget(view)
        self.banner=Q.QLabel('Multiple TIFFs, one FOV, shared ROIs. Use single-plane, single-channel data.'); self.banner.setWordWrap(True); vl.addWidget(self.banner)
        self.fov_trace_splitter=Q.QSplitter(QtCore.Qt.Vertical); self.fov_trace_splitter.setObjectName('FovTraceSplitter')
        self.fov_trace_splitter.setHandleWidth(5); self.fov_trace_splitter.setChildrenCollapsible(False); vl.addWidget(self.fov_trace_splitter,1)
        fov_panel=Q.QWidget(); fvl=Q.QVBoxLayout(fov_panel); fvl.setContentsMargins(0,0,0,0); self.fov_trace_splitter.addWidget(fov_panel)
        viewerbar=Q.QHBoxLayout(); fvl.addLayout(viewerbar)
        self.background=Q.QComboBox(); self.background.addItems(['Registered mean','Registered std','SUPPORT mean','Registered movie'])
        self.background.currentIndexChanged.connect(self.show_background); viewerbar.addWidget(self.background)
        self.contrast=Q.QDoubleSpinBox(); self.contrast.setRange(90,100); self.contrast.setDecimals(1); self.contrast.setValue(99.5)
        self.contrast.setToolTip('Display upper intensity percentile; does not modify data')
        viewerbar.addWidget(Q.QLabel('Display percentile')); viewerbar.addWidget(self.contrast); self.contrast.valueChanged.connect(self.display_percentile_changed)
        viewerbar.addWidget(Q.QLabel('Movie temporal average'))
        self.movie_average=Q.QComboBox(); self.movie_average.addItems(['Raw (1 frame)','3 frames','5 frames','9 frames','15 frames'])
        self.movie_average.setCurrentIndex(2)
        self.movie_average.setToolTip('Display only: centered temporal mean. Registration, TIFF export, ROI detection, and traces remain frame-resolved.')
        self.movie_average.currentIndexChanged.connect(self.show_background); viewerbar.addWidget(self.movie_average)
        self.filter=Q.QComboBox(); self.filter.addItems(['All ROIs','Cells','Not cells','Unreviewed']); viewerbar.addWidget(self.filter); self.filter.currentIndexChanged.connect(self.refresh)
        self.labels=Q.QCheckBox('ROI numbers'); viewerbar.addWidget(self.labels); self.labels.toggled.connect(self.render)
        maskbar=Q.QHBoxLayout(); fvl.addLayout(maskbar)
        maskbar.addWidget(Q.QLabel('ROI display'))
        self.mask_style=Q.QComboBox(); self.mask_style.addItems(['Multicolor masks','Outlines','Masks + outlines'])
        maskbar.addWidget(self.mask_style); self.mask_style.currentIndexChanged.connect(self.render)
        maskbar.addWidget(Q.QLabel('Opacity'))
        self.mask_opacity=Q.QSlider(QtCore.Qt.Horizontal); self.mask_opacity.setRange(5,90); self.mask_opacity.setValue(35)
        maskbar.addWidget(self.mask_opacity,1); self.mask_opacity.valueChanged.connect(self.render)
        maskbar.addWidget(Q.QLabel('Valid area'))
        self.valid_style=Q.QComboBox(); self.valid_style.addItems(['Mask + border','Border only','Hidden'])
        self.valid_style.setToolTip('Common area containing valid pixels in every registered frame')
        self.valid_style.currentIndexChanged.connect(self.render_valid_area); maskbar.addWidget(self.valid_style)
        area_edit=Q.QPushButton('Edit area...'); area_edit.clicked.connect(self.edit_valid_area); maskbar.addWidget(area_edit)
        self.brightness_button=Q.QPushButton('Brightness / Contrast...')
        self.brightness_button.setMinimumHeight(34); self.brightness_button.clicked.connect(lambda:self.guard(self.open_contrast))
        self.brightness_button.setToolTip('Display-only brightness and contrast (Ctrl+Shift+C)')
        self.contrast_shortcut=Q.QShortcut(QtGui.QKeySequence('Ctrl+Shift+C'),self)
        self.contrast_shortcut.activated.connect(lambda:self.guard(self.open_contrast))
        maskbar.addWidget(self.brightness_button)
        self.graphics=pg.GraphicsLayoutWidget(); fvl.addWidget(self.graphics,1)
        self.ax=self.graphics.addPlot(); self.ax.hideButtons(); self.ax.setAspectLocked(True); self.ax.invertY(True); self.ax.hideAxis('left'); self.ax.hideAxis('bottom')
        self.image=pg.ImageItem(); self.valid_overlay=pg.ImageItem(); self.overlay=pg.ImageItem()
        self.ax.addItem(self.image); self.ax.addItem(self.valid_overlay); self.ax.addItem(self.overlay)
        self.graphics.scene().sigMouseClicked.connect(self.click); self.texts=[]
        self.delete_shortcut=Q.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete),self)
        self.delete_shortcut.setAutoRepeat(False)
        self.delete_shortcut.activated.connect(lambda:self.guard(self.delete_from_keyboard))
        self.freehand_drawing=False
        self.graphics.viewport().installEventFilter(self)
        self.graphics.installEventFilter(self)
        self.fit_fov_button=Q.QToolButton(self.graphics.viewport())
        self.fit_fov_button.setFixedSize(30,30)
        self.fit_fov_button.setText('')
        self.fit_fov_button.setAccessibleName('Fit FOV')
        self.fit_fov_button.setToolTip('Fit FOV')
        self.fit_fov_button.setFocusPolicy(QtCore.Qt.NoFocus)
        self.fit_fov_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.fit_fov_button.setStyleSheet('QToolButton {background: #ffd400; border: 1px solid #9b8000; border-radius: 15px;} QToolButton:hover {background: #ffe866;} QToolButton:pressed {background: #caa800;}')
        self.fit_fov_button.clicked.connect(lambda:self.ax.autoRange())
        self.place_fit_fov_button()
        self.graphics.setMouseTracking(True); self.graphics.viewport().setMouseTracking(True)
        self.polyline=pg.PlotDataItem(pen=pg.mkPen('#ffd166',width=2),symbol='o'); self.ax.addItem(self.polyline)
        self.closure_ring=pg.ScatterPlotItem(pxMode=False,pen=pg.mkPen('#ff4545',width=2),brush=None)
        self.closure_ring.setZValue(30); self.ax.addItem(self.closure_ring)
        self.slider=Q.QSlider(QtCore.Qt.Horizontal); self.slider.setRange(0,0); self.slider.valueChanged.connect(self.frame_changed); fvl.addWidget(self.slider)
        self.frame_label=Q.QLabel(''); fvl.addWidget(self.frame_label)
        trace_panel=Q.QWidget(); tvl=Q.QVBoxLayout(trace_panel); tvl.setContentsMargins(0,0,0,0); self.fov_trace_splitter.addWidget(trace_panel)
        tracebar=Q.QHBoxLayout(); tvl.addLayout(tracebar)
        tracebar.addWidget(Q.QLabel('Traces'))
        self.trace_checks={}
        for key,label,color in [('F','Raw F','#65df9b'),('Fneu','Neuropil','#ff9c6b'),
                                ('spks','Deconv / spks','#d49aff'),('Fc','Corrected Fc','#6dcfff'),('dff','dF/F','#ffff8c')]:
            check=Q.QCheckBox(label); check.setChecked(key in ('F','Fneu','spks'))
            check.setStyleSheet('color: '+color); tracebar.addWidget(check); self.trace_checks[key]=(check,color)
            check.toggled.connect(self.plot_trace)
        self.normalize=Q.QCheckBox('Normalize display (0-1)'); tracebar.addWidget(self.normalize)
        self.normalize.toggled.connect(self.plot_trace)
        self.show_stim=Q.QCheckBox('Stim TTLs'); self.show_stim.setChecked(True); self.show_stim.setStyleSheet('color: #ff5a5f')
        self.show_stim.setToolTip('Red dashed lines and shaded intervals: stimulus TTL rising and falling edges')
        self.show_stim.toggled.connect(self.plot_trace); tracebar.addWidget(self.show_stim); tracebar.addStretch()
        self.stack_traces=Q.QCheckBox('Stack selected ROIs'); tracebar.addWidget(self.stack_traces)
        self.stack_traces.setToolTip('Ctrl/Shift-select rows in the ROIs table. Each selected ROI gets its own plot with a linked time axis.')
        self.export_roi_button=self.button(tracebar,'Export selected ROI(s)...',self.export_selected_roi)
        self.trace_plot=pg.PlotWidget(title='Select an ROI to inspect traces')
        self.trace_plot.setMinimumHeight(90)
        self.trace_scroll=Q.QScrollArea(); self.trace_scroll.setWidgetResizable(True)
        host=Q.QWidget(); self.trace_rows=Q.QVBoxLayout(host)
        self.trace_rows.setContentsMargins(0,0,0,0); self.trace_rows.setSpacing(2)
        self.trace_scroll.setWidget(host); tvl.addWidget(self.trace_scroll,1)
        self.trace_rows.addWidget(self.trace_plot,1)
        self.trace_widgets=[self.trace_plot]; self.frame_lines=[]
        self.stack_traces.toggled.connect(self.plot_trace)
        self.table.itemSelectionChanged.connect(lambda: self.plot_trace() if self.stack_traces.isChecked() else None)
        self.fov_trace_splitter.setStretchFactor(0,1); self.fov_trace_splitter.setStretchFactor(1,0)
        self.fov_trace_splitter.setSizes([720,180])
        self.stage_timer=QtCore.QTimer(self); self.stage_timer.timeout.connect(self.update_elapsed)
        split.setSizes([370,1130]); split.setStretchFactor(0,0); split.setStretchFactor(1,1)
        self.setStyleSheet('''QWidget {font-family: Segoe UI; font-size: 12px; background: #292929; color: #e4e4e4;}
            QPushButton {padding: 6px 12px; background: #414141; border: 1px solid #606060; border-radius: 3px;}
            QPushButton:hover {background: #535353;} QPushButton:disabled {color: #777;}
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {padding: 4px; background: #363636;}
            QTableWidget, QPlainTextEdit {background: #1c1c1c; gridline-color: #444;}
            QHeaderView::section {background: #414141; padding: 5px;}
            QTabWidget::pane {border: 1px solid #555;} QTabBar::tab {background: #383838; padding: 6px 13px;}
            QTabBar::tab:selected {background: #276b91; color: white;} QGroupBox {border: 1px solid #555; margin-top: 8px; padding-top: 8px;}
            QSplitter#FovTraceSplitter::handle:vertical {height: 5px; background: #606060; margin: 1px 0px;}
            QSplitter#FovTraceSplitter::handle:vertical:hover {background: #3689b8;}
            QProgressBar {border: 1px solid #555; text-align: center;} QProgressBar::chunk {background: #3689b8;}''')
    def button(self,parent,title,fn):
        b=Q.QPushButton(title); b.clicked.connect(lambda:self.guard(fn)); parent.addWidget(b); self.controls.append(b); return b
    def guard(self,fn):
        try: fn()
        except Exception:
            detail=traceback.format_exc(); path=self.step_log_path()
            try: path=write_diagnostic_log(detail,path)
            except OSError: path=None
            self.error(error_summary(detail),str(path) if path else '')
    def step_log_path(self):
        return B.STATE/'logs'/(time.strftime('%Y%m%d-%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.log')
    def error(self,message,path=''):
        message=error_summary(message)
        self.log.appendPlainText(message+(' | Log: '+path if path else ''))
        box=Q.QDialog(self); box.setWindowTitle('Action required'); box.setMinimumWidth(560)
        layout=Q.QVBoxLayout(box)
        summary=Q.QLabel(message); summary.setWordWrap(True); summary.setTextFormat(QtCore.Qt.PlainText)
        summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse|QtCore.Qt.TextSelectableByKeyboard)
        layout.addWidget(summary)
        if path:
            layout.addWidget(Q.QLabel('Diagnostic log:'))
            filename=Q.QLineEdit(path); filename.setReadOnly(True); layout.addWidget(filename)
        else:
            layout.addWidget(Q.QLabel('A diagnostic log could not be created.'))
        buttons=Q.QHBoxLayout(); layout.addLayout(buttons)
        if path:
            open_log=Q.QPushButton('Open log'); buttons.addWidget(open_log)
            open_log.clicked.connect(lambda:QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path)))
            copy_log=Q.QPushButton('Copy log path'); buttons.addWidget(copy_log)
            copy_log.clicked.connect(lambda:Q.QApplication.clipboard().setText(path))
        buttons.addStretch(); close=Q.QPushButton('OK'); close.setDefault(True)
        close.clicked.connect(box.accept); buttons.addWidget(close)
        box.exec_()
    def busy(self,value):
        for c in self.controls: c.setEnabled(not value)
        self.stop_button.setEnabled(value)
        self.table.setEnabled(not value); self.progress.setRange(0,0 if value else 1)
        if not value: self.progress.setValue(1)
    def stop_step(self):
        if self.worker is None or not self.worker.isRunning(): return
        self.worker.cancel.set(); self.stop_button.setEnabled(False)
        self.banner.setText('Stopping current step...')
    def step_stopped(self):
        self.banner.setText('Step stopped. Adjust settings and run it again.')
        self.log.appendPlainText('Stopped. Log: '+str(self.worker.log_path))
    def run(self,task,done):
        if self.worker and self.worker.isRunning(): return
        path=self.step_log_path()
        self.started=time.perf_counter(); self.stage_timer.start(500)
        self.worker=Worker(task,path); self.worker.log.connect(self.log_line)
        self.worker.failed.connect(self.error); self.worker.stopped.connect(self.step_stopped)
        self.worker.done.connect(lambda v:self.guard(lambda:done(v)))
        self.worker.finished.connect(self.stage_finished)
        self.busy(True); self.worker.start()
    def log_line(self,text):
        self.log.appendPlainText(text)
        if 'DEVICE:' in text: self.device_label.setText(text)
    def update_elapsed(self):
        self.elapsed_label.setText(f'Elapsed: {time.perf_counter()-self.started:.1f} s')
    def stage_finished(self):
        self.stage_timer.stop(); self.update_elapsed(); self.busy(False)
        self.total_seconds+=time.perf_counter()-self.started
        self.elapsed_label.setText(self.elapsed_label.text()+f' | Total work: {self.total_seconds:.1f} s')
        self.log.appendPlainText(self.elapsed_label.text())
    def update_sources(self):
        meta=json.loads((Path(self.session['root'])/'source.json').read_text(encoding='utf-8'))
        self.source_table.setRowCount(len(meta['files']))
        counts=np.bincount(np.asarray(meta['frame_map'])[:,1])
        for row,path in enumerate(meta['files']):
            for col,value in enumerate([str(row+1),str(counts[row+1]),path]):
                item=Q.QTableWidgetItem(value); item.setToolTip(path); self.source_table.setItem(row,col,item)
        self.source_table.setColumnWidth(0,48); self.source_table.setColumnWidth(1,65)
        self.source_summary.setText(f"{len(meta['files'])} TIFFs | {meta['shape'][0]} frames\nOutput: {self.session['root']}")
        self.left_tabs.setCurrentWidget(self.source_widget)
    def choose_profile(self):
        path,_=Q.QFileDialog.getOpenFileName(self,'Select PROFILE 2D JSON',
            self.cfg.get('profile_config') or str(Path(self.cfg['profile_root'])/'configs'),'PROFILE config (*.json)')
        if path:
            self.cfg['profile_config']=path; self.cfg['profile_checkpoint']=''
            self.profile_label.setText('PROFILE: '+path); self.profile_label.setToolTip(path)
    def config(self):
        self.cfg['suite2p_backend']='gpu' if self.s2p_backend.currentIndex() else 'cpu'
        self.cfg['registration']=['suite2p','matlab','already'][self.reg.currentIndex()]
        self.cfg['support']=self.support.isChecked(); self.cfg['continuous_files']=self.continuous.isChecked()
        self.cfg['registered_tiff_mode']=('separate','combined','none')[self.registered_tiff.currentIndex()]
        return copy.deepcopy(self.cfg)
    def reg_changed(self): self.prep_changed()
    def prep_changed(self):
        if self.session['ops_file']:
            self.banner.setText('Preparation settings changed. Run step 1 again; old results remain available for inspection.')
    def require_prepared(self):
        if not self.session['ops_file']: raise ValueError('Run step 1 first.')
        prep=B.relocate_config(json.loads((Path(self.session['ops_file']).parent/'preparation.json').read_text(encoding='utf-8'))['cfg'])
        cfg=self.config()
        keys=['registration','support','continuous_files','block_size','maxregshift','maxregshiftNR',
              'support_checkpoint','support_bs','reg_code']
        if any(prep.get(k,True if k=='continuous_files' else None)!=cfg.get(k) for k in keys):
            raise ValueError('Registration or denoising settings changed. Run step 1 again.')
    def settings(self):
        dialog=Settings(self.config(),self)
        if dialog.exec_():
            values=dialog.values()
            if values['block_size']%2: raise ValueError('Nonrigid block size must be even.')
            trace_keys=('fs','tau','neucoeff','inner_neuropil_radius','min_neuropil_pixels','circular_neuropil','win_baseline','sig_baseline')
            changed=any(self.cfg.get(k)!=values.get(k,self.cfg.get(k)) for k in trace_keys)
            sync_keys=('sync_variable','sync_frame_column','sync_stim_column','sync_sample_rate','sync_alignment','fs')
            sync_changed=any(self.cfg.get(k)!=values.get(k,self.cfg.get(k)) for k in sync_keys); self.cfg.update(values)
            self.param_label.setText(f'{self.cfg["fs"]:g} Hz  |  tau {self.cfg["tau"]:g} s')
            self.profile_label.setText('PROFILE: '+(self.cfg.get('profile_config') or 'no config selected'))
            if changed: self.invalidate(); self.banner.setText('Settings updated. Recompute traces; rerun step 1 if preparation settings changed.')
            if sync_changed and self.session.get('sync'):
                self.session['sync']=None; self.session['frame_times']=None; self.plot_trace()
                self.banner.setText('Synchronization settings changed. Import the synchronization MAT again.')
    def replace_ok(self):
        return not self.session['root'] or Q.QMessageBox.question(self,'Change dataset','Switch dataset? Save your session first if you want to retain the current ROI edits.')==Q.QMessageBox.Yes
    def import_files(self):
        paths,_=Q.QFileDialog.getOpenFileNames(self,'Select TIFF stacks from the same FOV','','TIFF (*.tif *.tiff)')
        if paths: self.import_paths(paths)
    def import_folder(self):
        folder=Q.QFileDialog.getExistingDirectory(self,'Select a TIFF folder from the same FOV')
        if folder: self.import_paths(self.tiffs_in_folder(folder))
    def tiffs_in_folder(self,folder):
        files=sorted((p for p in Path(folder).iterdir()
                      if p.is_file() and p.suffix.lower() in ('.tif','.tiff')),key=B.natural)
        generated=lambda p: bool(re.search(r'(?i)(?:^registered|_registered(?:_\d{8}-\d{6}(?:_\d+)?)?)$',p.stem))
        originals=[p for p in files if not generated(p)]
        if originals and len(originals)!=len(files):
            self.log.appendPlainText('Folder import: generated Registered TIFFs excluded; import these files explicitly if needed.')
        return originals or files
    def import_paths(self,paths):
        if not paths: raise ValueError('No TIFF files found.')
        if not self.replace_ok(): return
        dialog=ImportStacksDialog(paths,self)
        if not dialog.exec_(): return
        ordered=dialog.paths()
        preferences=QtCore.QSettings(str(B.STATE/'preferences.ini'),QtCore.QSettings.IniFormat)
        previous=preferences.value('import/last_output_parent','',type=str)
        if previous and Path(previous).is_dir():
            start=previous
        elif self.session.get('root') and Path(self.session['root']).parent.is_dir():
            start=str(Path(self.session['root']).parent)
        else:
            start=str(Path(ordered[0]).parent)
        destination=Q.QFileDialog.getExistingDirectory(self,'Select parent output directory',start)
        if not destination: return
        # Keep the session directory readable while still preventing collisions
        # when the same FOV is imported more than once in a minute.
        parent=Path(destination)
        base=parent/('TwoPhoton_'+time.strftime('%Y%m%d-%H%M'))
        session_dir=base; duplicate=2
        while session_dir.exists():
            session_dir=parent/(base.name+f'_{duplicate:02d}')
            duplicate+=1
        root=str(session_dir)
        def done(meta):
            preferences.setValue('import/last_output_parent',str(parent))
            preferences.sync()
            self.activate_imported_session(root,meta)
        self.run(partial(B.import_tiffs,ordered,root,preserve_order=True),done)
    def change_output_folder(self):
        if self.worker and self.worker.isRunning(): return
        if self.adding or getattr(self,'area_dialog',None) is not None:
            raise ValueError('Finish manual Add or close Edit area before changing the output folder.')
        preferences=QtCore.QSettings(str(B.STATE/'preferences.ini'),QtCore.QSettings.IniFormat)
        previous=preferences.value('import/last_output_parent',str(Path.home()),type=str)
        current=Path(self.session['root']).resolve() if self.session.get('root') else None
        initial=str(current.parent) if current else previous
        parent=Q.QFileDialog.getExistingDirectory(self,'Select new output parent folder',initial)
        if not parent: return
        parent=Path(parent).resolve()
        if current is None:
            preferences.setValue('import/last_output_parent',str(parent)); preferences.sync()
            self.source_summary.setText('No TIFF files imported\nNext output folder: '+str(parent))
            self.banner.setText('Default output parent changed: '+str(parent))
            return
        if parent==current.parent:
            self.banner.setText('Output folder is unchanged.'); return
        if current==parent or current in parent.parents:
            raise ValueError('Output relocation requires a destination outside the current session folder.')
        destination=parent/current.name; number=2
        while destination.exists():
            destination=parent/(current.name+f'_{number:02d}'); number+=1
        dialog=Q.QDialog(self); dialog.setWindowTitle('Change output folder'); dialog.resize(600,290)
        layout=Q.QVBoxLayout(dialog)
        note=Q.QLabel('Copy this session and its analysis files to the new folder, verify the copy, and switch future output there.\n'
                      'The original folder is retained. External raw TIFFs and existing Registered TIFFs beside them stay at their current paths.\n'
                      'Existing MAT/FIG/PNG/PDF exports are copied as-is. Newly exported results use the updated session paths.')
        note.setWordWrap(True); layout.addWidget(note)
        layout.addWidget(Q.QLabel('New session output:'))
        path=Q.QLineEdit(str(destination)); path.setReadOnly(True); layout.addWidget(path)
        buttons=Q.QDialogButtonBox(Q.QDialogButtonBox.Ok|Q.QDialogButtonBox.Cancel)
        buttons.button(Q.QDialogButtonBox.Ok).setText('Copy and switch output')
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); layout.addWidget(buttons)
        if not dialog.exec_(): return
        snapshot=dict(session=copy.deepcopy(self.session),cfg=self.config(),dirty=self.dirty)
        frame=self.slider.value(); levels=dict(self.display_levels)
        from output_location import relocate_output
        def done(state):
            self.session=state['session']; self.cfg=state['cfg']; self.dirty=state['dirty']
            self.movie=None; self.traces={}
            if self.session.get('ops_file'): self.load_movie()
            if not self.dirty and self.session.get('result'): self.load_traces()
            self.display_levels=levels; self.show_background(); self.slider.setValue(frame)
            self.update_sources(); self.refresh()
            preferences.setValue('import/last_output_parent',str(parent)); preferences.sync()
            self.banner.setText('Output folder changed. Existing analysis is ready; no recomputation is required.')
            self.log.appendPlainText('Current output: '+self.session['root'])
            self.log.appendPlainText('Saved session: '+str(Path(self.session['root'])/'session.npy'))
            self.log.appendPlainText('Original output retained: '+str(current))
        self.run(partial(relocate_output,snapshot,str(destination)),done)

    def dragEnterEvent(self,event):
        if self.worker and self.worker.isRunning(): event.ignore(); return
        if event.mimeData().hasUrls():
            paths=[Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
            if paths and all(path.is_dir() or path.suffix.lower() in ('.npy','.tif','.tiff') for path in paths):
                event.setDropAction(QtCore.Qt.CopyAction); event.accept(); return
        event.ignore()

    def dropEvent(self,event):
        if self.worker and self.worker.isRunning(): event.ignore(); return
        if not event.mimeData().hasUrls(): event.ignore(); return
        urls=event.mimeData().urls()
        if not all(url.isLocalFile() for url in urls): event.ignore(); return
        paths=[url.toLocalFile() for url in urls]
        event.setDropAction(QtCore.Qt.CopyAction); event.accept()
        self.guard(lambda:self.open_dropped_paths(paths))

    def open_dropped_paths(self,paths):
        if self.worker and self.worker.isRunning(): return
        sessions=[]; tiffs=[]; known=set()
        for item in paths:
            path=Path(item).resolve()
            if path.is_dir():
                saved=path/'session.npy'
                if saved.is_file(): sessions.append(saved); continue
                files=self.tiffs_in_folder(path)
                if not files: raise ValueError('Dropped folder contains no session.npy or TIFF stacks in its top level.')
            elif path.suffix.lower()=='.npy': sessions.append(path); continue
            elif path.is_file() and path.suffix.lower() in ('.tif','.tiff'): files=[path]
            else: raise ValueError('Drop one saved session.npy, a TIFF stack, or a folder of TIFF stacks.')
            for file in files:
                key=os.path.normcase(str(file))
                if key not in known: tiffs.append(file); known.add(key)
        if sessions:
            if len(sessions)!=1 or tiffs:
                raise ValueError('Drop only one session.npy at a time; do not mix sessions and TIFF stacks.')
            self.load_session_path(str(sessions[0]))
        elif tiffs:
            self.import_paths(tiffs)

    def show_error_reference(self):
        dialog=Q.QDialog(self); dialog.setWindowTitle('Common errors'); dialog.resize(720,560)
        layout=Q.QVBoxLayout(dialog); edit=Q.QPlainTextEdit(); edit.setReadOnly(True)
        from error_catalog import catalogue_text
        edit.setPlainText(catalogue_text()); layout.addWidget(edit)
        close=Q.QPushButton('Close'); close.clicked.connect(dialog.accept); layout.addWidget(close)
        dialog.exec_()
    def add_source_tiffs(self):
        if self.worker and self.worker.isRunning(): return
        if not self.session.get('root'):
            self.import_files(); return
        if self.adding: raise ValueError('Finish manual Add before appending TIFF stacks.')
        original=json.loads((Path(self.session['root'])/'source.json').read_text(encoding='utf-8'))
        paths,_=Q.QFileDialog.getOpenFileNames(self,'Add TIFF stacks to this FOV',
                          str(Path(original['files'][-1]).parent),'TIFF (*.tif *.tiff)')
        if not paths: return
        known={os.path.normcase(str(Path(p).resolve())) for p in original['files']}
        additional=[str(Path(p).resolve()) for p in paths if os.path.normcase(str(Path(p).resolve())) not in known]
        if not additional: raise ValueError('All selected TIFF files are already in this session.')
        dialog=ImportStacksDialog(additional,self)
        dialog.setWindowTitle('Append TIFF stacks - create extended session')
        note=Q.QLabel('Existing stacks stay in their original order. New stacks are appended in the order below.\n'
                      'A new combined session will be created. Registration, ROI detection, traces and synchronization must be rerun.\n'
                      'Original results and current ROI edits are preserved in a parent-session snapshot.')
        note.setWordWrap(True); dialog.layout().insertWidget(0,note)
        dialog.findChild(Q.QDialogButtonBox).button(Q.QDialogButtonBox.Ok).setText('Create extended session')
        if not dialog.exec_(): return
        combined=original['files']+dialog.paths()
        parent=Path(self.session['root']).parent
        base=parent/('TwoPhoton_'+time.strftime('%Y%m%d-%H%M')+'_extended')
        destination=base; number=2
        while destination.exists():
            destination=base.with_name(base.name+f'_{number:02d}'); number+=1
        from extend_session import create_extended_session
        snapshot=copy.deepcopy(self.session); config=self.config(); dirty=self.dirty
        def done(meta):
            self.activate_imported_session(str(destination),meta)
            self.banner.setText('Extended session created. Run steps 1-3 and import sync again; original analysis is preserved.')
            self.log.appendPlainText('Original analysis snapshot: '+str(destination/'parent_session.npy'))
        self.run(partial(create_extended_session,snapshot,config,dirty,combined,str(destination)),done)
    def activate_imported_session(self,root,meta):
        self.session=dict(root=root,ops_file='',stat=[],result='',frame_times=None,sync=None)
        self.undo_rois=None; self.edge_cache={}; self.update_sources()
        self.adding=False; self.polygon=[]; self.polyline.setData([],[])
        self.display_levels.clear()
        self.mean=None; self.movie=None; self.valid_area=None; self.valid_bounds=None; self.selected=-1
        self.source_meta=meta
        self.slider.blockSignals(True); self.slider.setRange(0,0); self.slider.blockSignals(False)
        self.frame_label.setText('Not registered. Run step 1.')
        self.invalidate(); self.refresh(); self.image.clear(); self.valid_overlay.clear(); self.overlay.clear()
        self.banner.setText(f"Imported {len(meta['files'])} TIFF files / {meta['shape'][0]} frames. Run step 1.")
        self.log.appendPlainText('Output: '+root)

    def prepare(self):
        if not self.session['root']: raise ValueError('Import TIFF files first.')
        if self.session['stat'] and Q.QMessageBox.question(self,'Re-register','Re-registering clears the current ROIs and traces. Previous results on disk are retained. Continue?')!=Q.QMessageBox.Yes: return
        cfg=self.config()
        self.run(partial(B.prepare,self.session['root'],cfg),self.prepared)
    def prepared(self,path):
        self.session.update(ops_file=path,stat=[],result=''); self.selected=-1; self.adding=False
        self.polygon=[]; self.polyline.setData([],[]); self.invalidate(); self.load_movie()
        self.refresh(); self.banner.setText('Registration complete. Run automatic detection; additional detectors merge new candidates.')
    def load_movie(self):
        self.display_levels.clear()
        self.source_meta=json.loads((Path(self.session['root'])/'source.json').read_text(encoding='utf-8'))
        ops=np.load(self.session['ops_file'],allow_pickle=True).item(); self.mean=ops['meanImg']; self.movie=B.binary_movie(ops)
        self.valid_area=B.valid_mask(ops); self.valid_bounds=tuple(int(v) for pair in (ops['yrange'],ops['xrange']) for v in pair)
        self.slider.setRange(0,len(self.movie)-1); self.show_background(); self.ax.autoRange()
    def detect(self):
        self.require_prepared()
        if self.adding: raise ValueError('Finish manual Add first.')
        cfg=self.config(); method=self.detector.currentText().lower()
        replace=self.replace_detector.isChecked()
        if method=='profile':
            if not cfg.get('profile_config','').strip():
                path,_=Q.QFileDialog.getOpenFileName(self,'Select a PROFILE 2D config matching your imaging scale',
                    str(Path(cfg['profile_root'])/'configs'),'PROFILE config (*.json)')
                if not path: return
                self.cfg['profile_config']=path; cfg=self.config()
                self.profile_label.setText('PROFILE: '+path)
            from profile_config import resolve_profile_config
            resolve_profile_config(cfg)
        def done(candidates):
            existing=[s for s in self.session['stat'] if not(replace and s['source']==method)]
            stat,ignored=B.merge_candidates(existing,candidates,self.mean.shape[1],cfg['merge_iou'])
            if [s['roi_id'] for s in stat]==[s['roi_id'] for s in self.session['stat']]:
                self.banner.setText(f'No new ROIs; {ignored} duplicates omitted. Existing traces retained.'); return
            self.session['stat']=stat; self.invalidate(); self.refresh()
            self.left_tabs.setCurrentWidget(self.roi_widget)
            self.banner.setText(f'{len(stat)} ROIs; {ignored} duplicates omitted. Run step 3.')
        self.run(partial(B.detect,self.session['ops_file'],cfg,method),done)
    def invalidate(self):
        self.dirty=True; self.traces={}; self.roi_revision+=1
        for plot in self.trace_widgets: plot.clear()
    def calculate(self):
        self.require_prepared()
        if self.adding: raise ValueError('Use Finish Add + Compute.')
        cfg=self.config(); stat=copy.deepcopy(self.session['stat'])
        if self.session.get('frame_times') is not None:
            meta=json.loads((Path(self.session['root'])/'source.json').read_text(encoding='utf-8'))
            B.validate_times(self.session['frame_times'],meta,cfg)
        if not stat: raise ValueError('Detect or add ROIs first.')
        destination=str(Path(self.session['root'])/('traces_'+uuid.uuid4().hex[:8]))
        self.run(partial(B.extract,self.session['ops_file'],stat,cfg,destination),self.calculated)
    def import_manual_neuropil(self):
        self.require_prepared()
        if self.dirty or not self.session.get('result'):
            raise ValueError('Compute traces first, or open the saved session matching Fall-2.mat.')
        path,_=Q.QFileDialog.getOpenFileName(self,'Select Fall-2.mat containing manualNeuropil',self.session['root'],'MAT files (*.mat)')
        if not path: return
        from studio_manual_neuropil import reextract_manual_neuropil
        destination=str(Path(self.session['root'])/('traces_manual_'+uuid.uuid4().hex[:8]))
        def done(result):
            self.calculated(result)
            self.save_session()
            self.banner.setText('Manual neuropil Fneu re-extracted. Fall-2.mat updated; original traces backed up.')
        self.run(partial(reextract_manual_neuropil,self.session['result'],path,destination,self.cfg['matlab']),done)
    def calculated(self,path):
        self.session['result']=path; self.session['stat']=list(np.load(Path(path)/'stat.npy',allow_pickle=True))
        self.dirty=False; self.load_traces(); self.refresh(); self.banner.setText('Traces ready. Click an ROI to inspect F, Fneu, Fc and spks. Review cell labels.')
        removed_path=Path(path)/'removed_roi_ids.json'
        if removed_path.exists():
            removed=json.loads(removed_path.read_text(encoding='utf-8'))
            if removed: self.banner.setText(f'Traces ready for {len(self.session["stat"])} ROIs; {len(removed)} invalid or redundant overlapping candidates removed. Review cell labels.')
    def load_traces(self):
        self.traces={n:np.load(Path(self.session['result'])/(n+'.npy'),mmap_mode='r') for n in ('F','Fneu','Fc','spks','dff')}
    def import_label_mask(self):
        self.require_prepared()
        path,_=Q.QFileDialog.getOpenFileName(self,'Import integer ROI label mask','','ROI masks (*.tif *.tiff *.npy *.npz)')
        if not path: return
        suffix=Path(path).suffix.lower()
        if suffix in ('.tif','.tiff'): labels=tifffile.imread(path)
        elif suffix=='.npy': labels=np.load(path,allow_pickle=False)
        else:
            with np.load(path,allow_pickle=False) as z:
                keys=list(z.files)
                preferred=next((key for key in ('labels','mask','masks') if key in z),None)
                if preferred is None:
                    if len(keys)!=1: raise ValueError('NPZ must contain labels/mask/masks or exactly one array.')
                    preferred=keys[0]
                labels=z[preferred]
        labels=np.asarray(labels)
        if labels.ndim==3:
            options=[f'Session {i+1} / plane {i+1}' for i in range(labels.shape[0])]
            selected,ok=Q.QInputDialog.getItem(self,'Select mask session',
                f'Mask contains {labels.shape[0]} planes. Choose the plane for this session:',options,0,False)
            if not ok: return
            labels=labels[options.index(selected)]
        if labels.ndim!=2 or tuple(labels.shape)!=tuple(self.mean.shape):
            raise ValueError(f'Label mask shape {labels.shape} does not match movie {self.mean.shape}.')
        if not np.issubdtype(labels.dtype,np.integer):
            rounded=np.rint(labels)
            if not np.allclose(labels,rounded,equal_nan=False): raise ValueError('Label mask must contain integer labels.')
            labels=rounded.astype(np.int64)
        ops=np.load(self.session['ops_file'],allow_pickle=True).item(); valid=B.valid_mask(ops)
        stat=[]
        for label in np.unique(labels):
            if int(label)==0: continue
            y,x=np.where(labels==label); inside=valid[y,x]; y,x=y[inside],x[inside]
            if len(y)<3: continue
            stat.append(dict(ypix=y,xpix=x,lam=np.ones(len(y),'float32'),source='external_mask',
                roi_id=f'mask_{int(label)}',external_label=int(label),accepted=True,reviewed=True,probability=np.nan))
        if not stat: raise ValueError('No valid nonzero ROI labels were found.')
        if self.session['stat']:
            answer=Q.QMessageBox.question(self,'Replace ROIs',f'Replace {len(self.session["stat"])} current ROIs with {len(stat)} imported ROIs?')
            if answer!=Q.QMessageBox.Yes: return
        self.session['stat']=stat; self.selected=-1; self.edge_cache={}; self.invalidate(); self.refresh()
        self.left_tabs.setCurrentWidget(self.roi_widget)
        self.banner.setText(f'Imported {len(stat)} label-mask ROIs. Run Compute traces.')
    def show_trace_heatmap(self):
        if self.dirty or not self.traces: raise ValueError('Compute traces first.')
        values=np.asarray(self.traces['dff'],dtype=float)
        center=np.nanmedian(values,axis=1,keepdims=True)
        scale=np.nanpercentile(np.abs(values-center),95,axis=1,keepdims=True)
        z=np.divide(values-center,scale,out=np.zeros_like(values),where=scale>1e-12)
        z=np.clip(z,-1,1)
        dialog=Q.QDialog(); dialog.setWindowTitle('dF/F trace heatmap | click a row to select ROI'); dialog.resize(1200,700)
        dialog.setModal(False)
        layout=Q.QVBoxLayout(dialog); graph=pg.GraphicsLayoutWidget(); layout.addWidget(graph,1)
        plot=graph.addPlot(); plot.setLabel('left','ROI'); plot.setLabel('bottom','Time','s')
        image=pg.ImageItem(z); n=z.shape[1]; times=self.session.get('frame_times')
        if times is None: times=np.arange(n,dtype=float)/self.cfg['fs']
        times=np.asarray(times,dtype=float).reshape(-1)
        start=float(times[0]) if len(times) else 0.
        duration=(float(times[-1]-times[0]) if len(times)>1 else (n-1)/self.cfg['fs'])
        image.setRect(QtCore.QRectF(start,0,max(duration,1e-9),z.shape[0])); plot.addItem(image)
        plot.getAxis('left').setTicks([[(i+.5,str(i+1)) for i in range(z.shape[0])]])
        highlight=pg.LinearRegionItem(values=(0,1),orientation='horizontal',movable=False,
            brush=pg.mkBrush(255,255,255,25),pen=pg.mkPen('#ff4545',width=2))
        highlight.setZValue(10); plot.addItem(highlight); highlight.hide()
        status=Q.QLabel('Click a heatmap row or select an ROI in the main image.'); layout.addWidget(status)
        revision=self.roi_revision
        def update_selection():
            valid=revision==self.roi_revision and not self.dirty
            if valid and 0<=self.selected<z.shape[0]:
                highlight.setRegion((self.selected,self.selected+1)); highlight.show()
                status.setText(f'Selected ROI {self.selected+1} | highlighted in the main image and trace panel')
            else:
                highlight.hide()
                if not valid: status.setText('ROI set changed: close this heatmap and reopen after Compute traces.')
        dialog.update_roi_selection=update_selection
        def choose_row(event):
            if event.button()!=QtCore.Qt.LeftButton or revision!=self.roi_revision or self.dirty: return
            if not plot.sceneBoundingRect().contains(event.scenePos()): return
            point=plot.vb.mapSceneToView(event.scenePos())
            index=int(np.floor(point.y()))
            if not (0<=index<z.shape[0] and start<=point.x()<=start+duration): return
            self.selected=index
            if index not in self.visible_ids(): self.filter.setCurrentIndex(0)
            self.table.clearSelection()
            for row in range(self.table.rowCount()):
                if self.table.item(row,0).data(QtCore.Qt.UserRole)==index:
                    self.table.selectRow(row); self.table.scrollToItem(self.table.item(row,0)); break
            self.render(); self.plot_trace()
        graph.scene().sigMouseClicked.connect(choose_row)
        # Parula-style anchors for the interactive preview; exported FIG uses MATLAB parula exactly.
        gradient=pg.ColorMap([0,.20,.40,.60,.80,1.0],[(53,42,135),(30,104,177),(18,158,174),(103,189,109),(230,190,49),(249,251,14)])
        image.setLookupTable(gradient.getLookupTable(0,1,256)); image.setLevels([-1,1])
        colorbar=pg.ColorBarItem(values=(-1,1),colorMap=gradient,label='Robust normalized dF/F',interactive=False)
        colorbar.setImageItem(image); graph.addItem(colorbar,row=0,col=1)
        buttons=Q.QHBoxLayout(); layout.addLayout(buttons); buttons.addStretch()
        export=Q.QPushButton('Export MATLAB FIG...'); buttons.addWidget(export)
        close=Q.QPushButton('Close'); close.clicked.connect(dialog.accept); buttons.addWidget(close)
        def export_fig():
            suggested=str(Path(self.session['root'])/'dff_trace_heatmap.fig')
            path,_=Q.QFileDialog.getSaveFileName(dialog,'Export editable MATLAB heatmap',suggested,'MATLAB figure (*.fig)')
            if not path: return
            destination=str(Path(path).with_suffix('.fig'))
            def done(saved):
                self.log.appendPlainText('Exported dF/F heatmap FIG: '+saved)
                self.banner.setText('Editable dF/F heatmap FIG exported with parula colorbar.')
            self.run(partial(B.export_dff_heatmap_fig,z,times,destination,matlab_executable=self.cfg['matlab']),done)
        export.clicked.connect(lambda:self.guard(export_fig))
        self.analysis_windows.append(dialog)
        dialog.finished.connect(lambda _=0:self.analysis_windows.remove(dialog) if dialog in self.analysis_windows else None)
        update_selection(); dialog.show()
    def visible_ids(self):
        mode=self.filter.currentIndex()
        return [i for i,s in enumerate(self.session['stat']) if mode==0 or mode==1 and s['accepted'] or mode==2 and not s['accepted'] or mode==3 and not s.get('reviewed',False)]
    def refresh(self):
        block=QtCore.QSignalBlocker(self.table)
        ids=self.visible_ids(); self.table.clearSelection(); self.table.setRowCount(len(ids))
        for row,i in enumerate(ids):
            s=self.session['stat'][i]; p=s.get('probability',np.nan) if s['source']=='suite2p' else np.nan
            values=[str(i+1),s['source'],('Cell' if s['accepted'] else 'Not cell')+('' if s.get('reviewed') else ' *'),f'{p:.2f}' if np.isfinite(p) else '—']
            for col,value in enumerate(values):
                item=Q.QTableWidgetItem(value); item.setData(QtCore.Qt.UserRole,i); self.table.setItem(row,col,item)
        del block
        self.table.resizeColumnsToContents(); self.render(); self.plot_trace()
    def open_contrast(self):
        if self.mean is None: raise ValueError('Register images or open a prepared session first.')
        if self.contrast_dialog is None:
            dialog=ContrastDialog(self); self.contrast_dialog=dialog
            dialog.levels_changed.connect(self.set_display_levels)
            dialog.auto_requested.connect(self.auto_display_levels)
            dialog.reset_requested.connect(self.reset_display_levels)
        self.show_background()
        self.contrast_dialog.show(); self.contrast_dialog.raise_(); self.contrast_dialog.activateWindow()
    def set_display_levels(self,lo,hi):
        self.display_levels[self.background.currentIndex()]=(lo,hi)
        self.image.setLevels((lo,hi))
    def auto_display_levels(self):
        if self.image.image is None: return
        lo,hi=B.imagej_auto_levels(self.image.image)
        self.set_display_levels(lo,hi)
        self.contrast_dialog.sync((lo,hi))
    def reset_display_levels(self):
        if self.image.image is None: return
        a=np.asarray(self.image.image); finite=a[np.isfinite(a)]
        if not finite.size: return
        lo,hi=float(finite.min()),float(finite.max())
        if hi<=lo: hi=lo+max(1.,abs(lo)*1e-6)
        self.set_display_levels(lo,hi); self.contrast_dialog.sync((lo,hi))
    def display_percentile_changed(self):
        self.display_levels.pop(self.background.currentIndex(),None); self.show_background()
    def show_background(self):
        if self.mean is None: return
        mode=self.background.currentIndex(); frame=self.mean
        folder=Path(self.session['ops_file']).parent
        if mode==1: frame=np.load(folder/'std.npy')
        if mode==2:
            if (folder/'denoised_mean.npy').exists(): frame=np.load(folder/'denoised_mean.npy')
            else: self.log.appendPlainText('SUPPORT was not used. Showing registered mean.')
        if mode==3:
            window=(1,3,5,9,15)[self.movie_average.currentIndex()]
            frame=B.temporal_average_frame(self.movie,self.slider.value(),window)
        mode=self.background.currentIndex()
        levels=self.display_levels.get(mode)
        if levels is None:
            finite=np.asarray(frame); finite=finite[np.isfinite(finite)]
            if not finite.size: return
            low,high=np.percentile(finite,[1,self.contrast.value()])
            levels=(float(low),float(max(high,low+max(abs(low)*1e-8,1e-7))))
        self.image.setImage(frame,levels=levels,autoLevels=False)
        if self.contrast_dialog is not None:
            self.contrast_dialog.update_image(frame,levels)
        self.frame_changed_label()
    def frame_changed(self):
        if self.background.currentIndex()==3: self.show_background()
        self.frame_changed_label()
        for line in self.frame_lines: line.setValue(self.current_frame_time())
    def current_frame_time(self):
        t=self.slider.value(); times=self.session.get('frame_times')
        return float(t/self.cfg['fs'] if times is None else times[t])
    def frame_changed_label(self):
        if self.movie is None: return
        t=self.slider.value(); times=self.session.get('frame_times')
        seconds=t/self.cfg['fs'] if times is None else times[t]
        source='nominal time' if times is None else 'acquisition time'
        _,file_id,page=self.source_meta['frame_map'][t]
        filename=Path(self.source_meta['files'][file_id-1]).name
        self.frame_label.setText(f'Frame {t+1}/{len(self.movie)} | {seconds:.3f} s ({source}) | {filename} / page {page}')
    def render(self):
        for window in self.analysis_windows:
            if hasattr(window,'update_roi_selection'): window.update_roi_selection()
        if self.mean is None: return
        self.render_valid_area()
        overlay=np.zeros((*self.mean.shape,4),dtype='uint8')
        for item in self.texts: self.ax.removeItem(item)
        self.texts=[]
        for i in self.visible_ids():
            s=self.session['stat'][i]; y,x=s['ypix'],s['xpix']; key=s['roi_id']
            rgb=tuple(int(round(v*255)) for v in B.roi_color(key))
            if i==self.selected: rgb=(255,220,65)
            color=(*rgb,255)
            if self.mask_style.currentIndex()!=1:
                overlay[y,x]=(*rgb,int(self.mask_opacity.value()*2.55))
            if self.mask_style.currentIndex()!=0 or i==self.selected:
                if key not in self.edge_cache:
                    y0,x0=int(y.min()),int(x.min()); local=np.zeros((int(y.max())-y0+1,int(x.max())-x0+1),bool)
                    local[y-y0,x-x0]=True; ey,ex=np.where(local&~binary_erosion(local))
                    self.edge_cache[key]=(ey+y0,ex+x0)
                ey,ex=self.edge_cache[key]; overlay[ey,ex]=color
            if self.labels.isChecked():
                label=pg.TextItem(str(i+1),color=color[:3],anchor=(.5,.5)); label.setPos(float(np.median(s['xpix'])),float(np.median(s['ypix']))); self.ax.addItem(label); self.texts.append(label)
        self.overlay.setImage(overlay,autoLevels=False)
    def edit_valid_area(self):
        if self.worker and self.worker.isRunning(): raise ValueError('Wait for processing to finish before editing the valid area.')
        self.require_prepared()
        if self.adding: raise ValueError('Finish manual Add before editing the valid area.')
        if getattr(self,'area_dialog',None) is not None:
            self.area_dialog.raise_(); return
        ops_path=self.session['ops_file']; ops=np.load(ops_path,allow_pickle=True).item()
        h,w=int(ops['Ly']),int(ops['Lx']); y0,y1,x0,x1=self.valid_bounds
        dialog=Q.QDialog(self); self.area_dialog=dialog
        dialog.setWindowTitle('Edit valid registration area'); dialog.setWindowFlags(dialog.windowFlags()|QtCore.Qt.Tool)
        root=Q.QVBoxLayout(dialog)
        note=Q.QLabel('Drag or resize the cyan rectangle in the FOV, or enter edge margins.\nThis changes the ROI analysis area. Registered TIFF pixels are preserved.\nAreas beyond Auto may contain wrapped or padded pixels in some frames.')
        note.setWordWrap(True); root.addWidget(note)
        form=Q.QFormLayout(); root.addLayout(form); fields=[]
        for label,maximum,value in [('Top (px)',h-1,y0),('Bottom (px)',h-1,h-y1),
                                    ('Left (px)',w-1,x0),('Right (px)',w-1,w-x1)]:
            field=Q.QSpinBox(); field.setRange(0,maximum); field.setValue(value)
            fields.append(field); form.addRow(label,field)
        roi=pg.RectROI([x0,y0],[x1-x0,y1-y0],pen=pg.mkPen('#00e5ff',width=2),
                       maxBounds=QtCore.QRectF(0,0,w,h))
        roi.setZValue(100); self.ax.addItem(roi); busy=[False]; mode=['manual']
        def fields_to_roi():
            if busy[0]: return
            top,bottom,left,right=[f.value() for f in fields]
            if top+bottom>=h or left+right>=w: return
            busy[0]=True; mode[0]='manual'
            roi.setPos([left,top]); roi.setSize([w-left-right,h-top-bottom]); busy[0]=False
        def roi_to_fields():
            if busy[0]: return
            busy[0]=True; mode[0]='manual'
            pos=roi.pos(); size=roi.size()
            x=max(0,min(w-1,int(round(pos.x())))); y=max(0,min(h-1,int(round(pos.y()))))
            xx=max(x+1,min(w,int(round(pos.x()+size.x())))); yy=max(y+1,min(h,int(round(pos.y()+size.y()))))
            for f,v in zip(fields,[y,h-yy,x,w-xx]): f.setValue(v)
            busy[0]=False
        for f in fields: f.valueChanged.connect(fields_to_roi)
        roi.sigRegionChanged.connect(roi_to_fields)
        def preset(bounds,label):
            yy,yyy,xx,xxx=bounds; busy[0]=True
            for f,v in zip(fields,[yy,h-yyy,xx,w-xxx]): f.setValue(int(v))
            roi.setPos([xx,yy]); roi.setSize([xxx-xx,yyy-yy]); busy[0]=False; mode[0]=label
        actions=Q.QHBoxLayout(); root.addLayout(actions)
        auto=Q.QPushButton('Auto from all-frame shifts'); actions.addWidget(auto)
        auto.clicked.connect(lambda:preset(B.automatic_valid_bounds(ops),'auto'))
        full=Q.QPushButton('Full FOV'); actions.addWidget(full)
        full.clicked.connect(lambda:preset([0,h,0,w],'manual'))
        buttons=Q.QDialogButtonBox(Q.QDialogButtonBox.Apply|Q.QDialogButtonBox.Cancel); root.addWidget(buttons)
        def apply():
            if self.worker and self.worker.isRunning():
                Q.QMessageBox.warning(dialog,'Processing','Wait for processing to finish before applying the area.'); return
            if self.session['ops_file']!=ops_path:
                Q.QMessageBox.warning(dialog,'Session changed','Reopen the valid area editor for this session.'); dialog.reject(); return
            top,bottom,left,right=[f.value() for f in fields]; bounds=[top,h-bottom,left,w-right]
            try: updated=B.set_valid_bounds(ops_path,bounds,mode[0])
            except Exception as exc: Q.QMessageBox.warning(dialog,'Invalid area',str(exc)); return
            self.valid_bounds=tuple(bounds); self.valid_area=B.valid_mask(updated)
            self.invalidate(); self.render()
            self.log.appendPlainText(f'Valid area updated ({mode[0]}): y=[{top},{h-bottom}), x=[{left},{w-right}). Recompute traces; rerun detection to recover previously excluded ROIs.')
            self.banner.setText('Valid area updated. Recompute traces; rerun detection to find ROIs in the expanded area.')
            dialog.accept()
        buttons.button(Q.QDialogButtonBox.Apply).clicked.connect(apply)
        buttons.rejected.connect(dialog.reject)
        def cleanup(*_):
            self.ax.removeItem(roi); self.area_dialog=None; dialog.deleteLater()
        dialog.finished.connect(cleanup); dialog.resize(520,310); dialog.show()

    def render_valid_area(self):
        if self.mean is None or self.valid_area is None: return
        mode=self.valid_style.currentIndex(); rgba=np.zeros((*self.mean.shape,4),dtype='uint8')
        if mode==2:
            self.valid_overlay.setImage(rgba,autoLevels=False); return
        if mode==0: rgba[~self.valid_area]=[0,0,0,180]
        y0,y1,x0,x1=self.valid_bounds; thickness=2
        color=np.asarray([255,205,0,255],dtype='uint8')
        rgba[y0:min(y0+thickness,y1),x0:x1]=color
        rgba[max(y0,y1-thickness):y1,x0:x1]=color
        rgba[y0:y1,x0:min(x0+thickness,x1)]=color
        rgba[y0:y1,max(x0,x1-thickness):x1]=color
        self.valid_overlay.setImage(rgba,autoLevels=False)
    def select_row(self,row,col):
        self.selected=self.table.item(row,0).data(QtCore.Qt.UserRole); self.render(); self.plot_trace()
    def eventFilter(self,watched,event):
        if hasattr(self,'graphics') and watched is self.graphics.viewport() and event.type()==QtCore.QEvent.Resize:
            self.place_fit_fov_button()
        if hasattr(self,'graphics') and watched in (self.graphics,self.graphics.viewport()):
            if event.type()==QtCore.QEvent.KeyPress and event.key()==QtCore.Qt.Key_Escape and (self.freehand_drawing or self.polygon):
                self.freehand_drawing=False; self.polygon=[]
                self.polyline.setData([],[]); self.closure_ring.clear()
                event.accept(); return True
        if hasattr(self,'graphics') and watched is self.graphics.viewport():
            kind=event.type()
            active=self.adding and self.shape.currentIndex()==1 and self.mean is not None
            if active and kind==QtCore.QEvent.ContextMenu: return True
            if active and kind==QtCore.QEvent.MouseButtonPress and event.button()==QtCore.Qt.RightButton:
                if self.worker and self.worker.isRunning(): return True
                if self.freehand_drawing:
                    self.freehand_drawing=False; self.closure_ring.clear()
                    self.guard(self.finish_polygon); return True
                point=self.ax.vb.mapSceneToView(self.graphics.mapToScene(event.pos()))
                if 0<=point.x()<self.mean.shape[1] and 0<=point.y()<self.mean.shape[0]:
                    self.freehand_drawing=True; self.polygon=[[point.x(),point.y()]]
                    self.freehand_left_start=False
                    self.graphics.viewport().setFocus()
                    self.polyline.setData([point.x()],[point.y()])
                    self.closure_ring.setData([point.x()],[point.y()],size=10.8)
                return True
            if self.freehand_drawing and not active:
                self.freehand_drawing=False; self.polygon=[]; self.polyline.setData([],[]); self.closure_ring.clear()
            if active and self.freehand_drawing and kind==QtCore.QEvent.MouseMove:
                point=self.ax.vb.mapSceneToView(self.graphics.mapToScene(event.pos()))
                if 0<=point.x()<self.mean.shape[1] and 0<=point.y()<self.mean.shape[0]:
                    last=self.polygon[-1]
                    if (point.x()-last[0])**2+(point.y()-last[1])**2>=0.25:
                        self.polygon.append([point.x(),point.y()]); xy=np.asarray(self.polygon)
                        self.polyline.setData(xy[:,0],xy[:,1])
                    # Cellpose is_at_start: image-pixel thresholds for brush_size=3.
                    # Check every mouse move, including moves smaller than sample spacing.
                    distance=(point.x()-self.polygon[0][0])**2+(point.y()-self.polygon[0][1])**2
                    if distance>max(6,3*3)**2: self.freehand_left_start=True
                    if self.freehand_left_start and len(self.polygon)>=5 and distance<max(3,3*1.8)**2:
                        self.freehand_drawing=False; self.closure_ring.clear()
                        self.guard(self.finish_polygon)
                return True
            if active and kind==QtCore.QEvent.MouseButtonRelease and event.button()==QtCore.Qt.RightButton:
                return True
            if kind==QtCore.QEvent.KeyPress and event.key()==QtCore.Qt.Key_Escape and self.freehand_drawing:
                self.freehand_drawing=False; self.polygon=[]; self.polyline.setData([],[]); self.closure_ring.clear(); return True
        return super().eventFilter(watched,event)
    def click(self,event):
        if getattr(self,'area_dialog',None) is not None: return
        if self.worker and self.worker.isRunning() or self.mean is None: return
        if not self.ax.sceneBoundingRect().contains(event.scenePos()): return
        point=self.ax.vb.mapSceneToView(event.scenePos()); x,y=int(np.floor(point.x())),int(np.floor(point.y()))
        if not (0<=y<self.mean.shape[0] and 0<=x<self.mean.shape[1]): return
        if self.adding:
            if self.shape.currentIndex()==1: return
            if event.button()!=QtCore.Qt.LeftButton: return
            if self.shape.currentIndex()==1:
                self.polygon.append([point.x(),point.y()]); xy=np.array(self.polygon); self.polyline.setData(xy[:,0],xy[:,1]); return
            yy,xx=np.ogrid[:self.mean.shape[0],:self.mean.shape[1]]
            self.guard(lambda:self.add_mask((xx-x)**2+(yy-y)**2<=self.radius.value()**2)); return
        for i in reversed(self.visible_ids()):
            s=self.session['stat'][i]
            if np.any((s['ypix']==y)&(s['xpix']==x)):
                if event.button()==QtCore.Qt.RightButton:
                    self.delete_indices([i],f'Right-click deleted ROI {i+1}.')
                    return
                if event.button()!=QtCore.Qt.LeftButton: return
                self.selected=i; self.table.clearSelection()
                for row in range(self.table.rowCount()):
                    if self.table.item(row,0).data(QtCore.Qt.UserRole)==i: self.table.selectRow(row); break
                self.render(); self.plot_trace(); return
    def plot_trace(self):
        ids=[]
        if self.stack_traces.isChecked():
            ids=sorted({self.table.item(row.row(),0).data(QtCore.Qt.UserRole)
                        for row in self.table.selectionModel().selectedRows()})
        if not ids: ids=[self.selected]
        ids=[i for i in ids if 0<=i<len(self.session['stat'])]
        count=max(1,len(ids))
        while len(self.trace_widgets)>count:
            plot=self.trace_widgets.pop(); plot.setXLink(None)
            self.trace_rows.removeWidget(plot); plot.hide(); plot.deleteLater()
        while len(self.trace_widgets)<count:
            plot=pg.PlotWidget(); plot.setXLink(self.trace_plot)
            self.trace_widgets.append(plot); self.trace_rows.addWidget(plot,1)
        self.frame_lines=[]
        sync=self.session.get('sync')
        n=self.traces['F'].shape[1] if not self.dirty and 'F' in self.traces else 0
        t=self.session.get('frame_times')
        if t is None and n: t=np.arange(n)/self.cfg['fs']
        for row,plot in enumerate(self.trace_widgets):
            plot.clear()
            plot.setMinimumHeight(120 if count>1 else 90)
            plot.showAxis('bottom',row==count-1); plot.setLabel('bottom','Time','s')
            line=pg.InfiniteLine(pos=self.current_frame_time() if self.movie is not None else 0.,angle=90,
                                 pen=pg.mkPen('#35c9ff',width=1.5),movable=False)
            line.setZValue(20); plot.addItem(line,ignoreBounds=True); self.frame_lines.append(line)
            if sync and self.show_stim.isChecked():
                pen=pg.mkPen('#ff5a5f',width=1,style=QtCore.Qt.DashLine)
                starts=np.asarray(sync.get('stim_times',[])).reshape(-1)
                ends=np.asarray(sync.get('stim_end_times',[])).reshape(-1)
                for j,onset in enumerate(starts):
                    if j<len(ends):
                        region=pg.LinearRegionItem(values=(float(onset),float(ends[j])),movable=False,
                                                   brush=pg.mkBrush(255,72,78,85),pen=pen)
                        region.setZValue(-20); plot.addItem(region,ignoreBounds=True)
                    plot.addItem(pg.InfiniteLine(pos=float(onset),angle=90,pen=pen,movable=False),ignoreBounds=True)
            if not ids:
                plot.setTitle(f'{len(sync.get("stim_frames",[]))} stimulus intervals loaded | Select an ROI' if sync
                              else 'Select an ROI to inspect traces')
                continue
            i=ids[row]
            if not n:
                plot.setTitle(f'ROI {i+1}: pending. Finish Add or run step 3.')
                continue
            for name,(check,color) in self.trace_checks.items():
                if not check.isChecked(): continue
                values=self.traces[name][i]
                if self.normalize.isChecked():
                    finite=values[np.isfinite(values)]
                    if len(finite): values=(values-finite.min())/max(float(np.ptp(finite)),1e-9)
                curve=plot.plot(t,values,pen=color,connect='finite')
                curve.setDownsampling(auto=True,method='peak'); curve.setClipToView(True)
            plot.setLabel('left','Normalized' if self.normalize.isChecked() else 'Native units / dF/F')
            plot.setTitle(f'ROI {i+1} | registered movie, without SUPPORT denoising')
        if not n and t is not None and len(t)>1:
            self.trace_plot.setXRange(float(t[0]),float(t[-1]),padding=0)
    def label(self,value):
        if self.selected<0: return
        self.session['stat'][self.selected].update(accepted=value,reviewed=True); self.refresh()
    def delete_from_keyboard(self):
        if self.worker and self.worker.isRunning(): return
        if self.freehand_drawing or self.polygon or getattr(self,'area_dialog',None) is not None: return
        focus=Q.QApplication.focusWidget()
        if focus is None: return
        if isinstance(focus,(Q.QLineEdit,Q.QAbstractSpinBox,Q.QTextEdit,Q.QPlainTextEdit,Q.QComboBox)): return
        # Delete text normally in text/parameter fields and restrict ROI deletion
        # to focus within the FOV or the ROI table.
        if not any(focus is area or area.isAncestorOf(focus) for area in (self.graphics,self.table)):
            return
        self.delete()
    def place_fit_fov_button(self):
        if not hasattr(self,'fit_fov_button'): return
        viewport=self.graphics.viewport()
        self.fit_fov_button.move(10,max(0,viewport.height()-40))
        self.fit_fov_button.raise_(); self.fit_fov_button.show()
    def delete(self):
        ids=sorted({self.table.item(row.row(),0).data(QtCore.Qt.UserRole) for row in self.table.selectionModel().selectedRows()},reverse=True)
        if not ids and self.selected>=0: ids=[self.selected]
        if not ids: return
        self.delete_indices(ids,f'Deleted {len(ids)} ROIs.')
    def delete_indices(self,ids,message):
        ids=sorted(set(ids),reverse=True)
        self.undo_rois=copy.deepcopy(self.session['stat'])
        for i in ids: del self.session['stat'][i]
        self.selected=-1; self.invalidate(); self.refresh()
        next_step=' Continue adding; Finish Add + Compute updates traces.' if self.adding else ' Recompute traces to update neuropil.'
        self.banner.setText(message+' Undo is available.'+next_step)
    def undo_delete(self):
        if self.undo_rois is None: return
        self.session['stat']=self.undo_rois; self.undo_rois=None; self.selected=-1; self.invalidate(); self.refresh()
        self.banner.setText('ROI deletion undone.'+(' Continue adding; Finish Add + Compute updates traces.' if self.adding else ' Recompute traces to update neuropil.'))
    def start_add(self):
        self.require_prepared(); self.adding=True; self.pending=0; self.polygon=[]
        self.banner.setText('ADD MODE: Polygon = right-click once, move around the cell, return to the start to auto-close. Esc cancels. Finish Add + Compute extracts traces.')
    def add_mask(self,mask):
        ops=np.load(self.session['ops_file'],allow_pickle=True).item()
        if np.any(mask&~B.valid_mask(ops)): raise ValueError('ROI crosses the common valid registration area.')
        y,x=np.where(mask)
        if len(y)<3: raise ValueError('ROI is too small.')
        s=dict(ypix=y,xpix=x,lam=np.ones(len(y),'float32'),source='manual',roi_id=uuid.uuid4().hex[:12],accepted=True,reviewed=True,probability=np.nan)
        merged,ignored=B.merge_candidates(self.session['stat'],[s],self.mean.shape[1],self.cfg['merge_iou'])
        if ignored: raise ValueError('Strong overlap with an existing ROI. Check for duplicates.')
        self.session['stat']=merged; self.selected=len(merged)-1; self.pending+=1; self.invalidate(); self.refresh()
    def finish_polygon(self):
        if not self.adding or len(self.polygon)<3: raise ValueError('Select at least three vertices in Add mode.')
        from matplotlib.path import Path as Polygon
        yy,xx=np.mgrid[:self.mean.shape[0],:self.mean.shape[1]]
        boundary=np.asarray(self.polygon,dtype=float)
        boundary=np.vstack([boundary,boundary[0]])
        self.add_mask(Polygon(boundary).contains_points(np.column_stack([xx.ravel()+.5,yy.ravel()+.5])).reshape(self.mean.shape))
        self.polygon=[]; self.polyline.setData([],[])
    def finish_add(self):
        if self.polygon: self.finish_polygon()
        self.adding=False; self.calculate()
    def save_session(self):
        if not self.session['root']: return
        path,_=Q.QFileDialog.getSaveFileName(self,'Save session',str(Path(self.session['root'])/'session.npy'),'Session (*.npy)')
        if path:
            state=dict(session=self.session,cfg=self.config(),dirty=self.dirty)
            with open(path,'wb') as f: np.save(f,state)
            self.log.appendPlainText('Session saved: '+path)
    def load_session(self):
        path,_=Q.QFileDialog.getOpenFileName(self,'Open a trusted session generated by this application','','Session (*.npy)')
        if path: self.load_session_path(path)
    def load_session_path(self,path):
        if self.worker and self.worker.isRunning(): return
        if not path or not self.replace_ok(): return
        loaded=np.load(path,allow_pickle=True)
        if not isinstance(loaded,np.ndarray) or loaded.size!=1:
            raise ValueError('The dropped NPY is not a saved TwoPhoton Studio session.')
        value=loaded.item()
        if not isinstance(value,dict) or not isinstance(value.get('session'),dict) or not isinstance(value.get('cfg'),dict):
            raise ValueError('The dropped NPY is not a saved TwoPhoton Studio session.')
        state=value['session']
        if not all(key in state for key in ('root','ops_file','stat','result')):
            raise ValueError('The dropped NPY is not a complete TwoPhoton Studio session.')
        if not (Path(state['root'])/'source.json').is_file():
            raise FileNotFoundError('Saved session source.json is unavailable: '+str(Path(state['root'])/'source.json'))
        for key in ('ops_file','result'):
            if state.get(key) and not Path(state[key]).exists():
                raise FileNotFoundError('Saved session '+key+' path is unavailable: '+str(state[key]))
        self.cfg=value['cfg']; self.session=state; self.dirty=value.get('dirty',True)
        # Legacy sessions used rectangular masks; retain their extraction setting.
        self.cfg.setdefault('circular_neuropil',False)
        self.cfg=B.relocate_config({**B.defaults(),**self.cfg}); self.s2p_backend.setCurrentIndex(1 if self.cfg.get('suite2p_backend')=='gpu' else 0)
        self.undo_rois=None; self.update_sources(); self.profile_label.setText('PROFILE: '+(self.cfg.get('profile_config') or 'no config selected'))
        self.param_label.setText(f'{self.cfg["fs"]:g} Hz  |  tau {self.cfg["tau"]:g} s')
        self.reg.blockSignals(True); self.reg.setCurrentIndex(['suite2p','matlab','already'].index(self.cfg['registration'])); self.reg.blockSignals(False)
        self.support.setChecked(self.cfg['support']); self.continuous.setChecked(self.cfg.get('continuous_files',True)); self.selected=-1; self.adding=False
        mode=self.cfg.get('registered_tiff_mode','separate')
        self.registered_tiff.setCurrentIndex(('separate','combined','none').index(mode) if mode in ('separate','combined','none') else 0)
        self.traces={}; self.mean=None; self.movie=None
        if self.session['ops_file']: self.load_movie()
        if not self.dirty and self.session['result']: self.load_traces()
        self.refresh(); self.banner.setText('Session loaded.')
    def import_times(self):
        if not self.session['root']: raise ValueError('Import images first.')
        paths,_=Q.QFileDialog.getOpenFileNames(self,'Select synchronization MAT file(s)', '', 'MAT synchronization (*.mat)')
        if paths:
            source=json.loads((Path(self.session['root'])/'source.json').read_text(encoding='utf-8'))
            sync=B.read_sync_files(paths,source,self.config())
            self.session['frame_times']=B.validate_times(sync['frame_times'],source,self.config())
            self.session['sync']=sync
            threshold_text=f'frame threshold {sync.get("frame_threshold",float("nan")):.4g} V, stimulus threshold {sync.get("stim_threshold",float("nan")):.4g} V, 10 ms debounce'
            terminal='; terminal frame pulse removed' if sync.get('trimmed_terminal_frame') else ''
            observed=len(sync['frame_ttl_samples']); expected=len(sync['frame_times'])
            delta=observed-expected
            discrepancy=(f'; {observed} observed frame TTL rises for {expected} TIFF frames (difference {delta:+d}); later frame times inferred from {self.cfg["fs"]:g} Hz'
                         if sync.get('alignment_mode')=='first_edge' else '')
            self.log.appendPlainText(f'Synchronization loaded: {observed} observed frame rising edges, {expected} TIFF frames, {len(sync["stim_frames"])} stimulus high intervals; {sync["sync_mode"]}; {threshold_text}{terminal}{discrepancy}')
            status=(f'First-rise anchor: {observed} observed TTLs for {expected} TIFF frames; remaining timestamps use nominal {self.cfg["fs"]:g} Hz.'
                    if sync.get('alignment_mode')=='first_edge' else f'Strict TTL match: {observed} frame rises for {expected} TIFF frames.')
            self.banner.setText(f'{status} {len(sync["stim_frames"])} stimulus intervals mapped. Red shading marks stimulus intervals.')
            self.plot_trace()
    def export(self):
        if self.dirty or not self.session['result']: raise ValueError('Compute traces for the current ROIs first.')
        path,_=Q.QFileDialog.getSaveFileName(self,'Export Fall.mat',str(Path(self.session['root'])/'Fall.mat'),
                                           'MATLAB (*.mat)',options=Q.QFileDialog.DontConfirmOverwrite)
        if not path: return
        if not Path(path).suffix: path+='.mat'
        replace=False
        if Path(path).exists():
            prompt=Q.QMessageBox(self); prompt.setWindowTitle('Replace existing Fall.mat?')
            prompt.setIcon(Q.QMessageBox.Question)
            prompt.setText('This file already exists. Replace it with the current analysis results?')
            prompt.setInformativeText(str(Path(path)))
            replace_button=prompt.addButton('Replace',Q.QMessageBox.AcceptRole)
            cancel_button=prompt.addButton(Q.QMessageBox.Cancel)
            prompt.setDefaultButton(cancel_button); prompt.setEscapeButton(cancel_button); prompt.exec_()
            if prompt.clickedButton()!=replace_button: return
            replace=True
        self.run(partial(B.export_mat,self.session,path,replace=replace),
                 lambda p:self.log.appendPlainText(('Replaced: ' if replace else 'Exported: ')+p))
    def export_selected_roi(self):
        if self.dirty or not self.session['result']: raise ValueError('Compute traces for the current ROI set before exporting selected ROIs.')
        indices=sorted({self.table.item(row.row(),0).data(QtCore.Qt.UserRole) for row in self.table.selectionModel().selectedRows()})
        if not indices and 0<=self.selected<len(self.session['stat']): indices=[self.selected]
        if not indices: raise ValueError('Select one or more ROIs in the ROI table first (Ctrl/Shift for multiple selection).')
        suggested=Path(self.session['root'])/('selected_ROIs.mat' if len(indices)>1 else f'ROI_{indices[0]+1}_{self.session["stat"][indices[0]].get("roi_id","")}.mat')
        path,_=Q.QFileDialog.getSaveFileName(self,'Export selected ROI bundle',str(suggested),'MATLAB bundle (*.mat)')
        if not path: return
        names=[name for name,(check,_) in self.trace_checks.items() if check.isChecked()]
        if not names: raise ValueError('Enable at least one trace for the PNG/PDF export.')
        normalized=self.normalize.isChecked(); show_stim=self.show_stim.isChecked(); percentile=self.contrast.value()
        def done(paths):
            self.log.appendPlainText('Selected ROI export:\n'+'\n'.join(paths.values()))
            self.banner.setText(f'{len(indices)} ROI(s) exported: MAT, editable trace FIG, trace PNG/PDF, and registered-mean mask PNG/PDF.')
        export_session=dict(self.session,cfg=self.config())
        self.run(partial(B.export_selected_rois,export_session,indices,path,names,normalized,show_stim,percentile),done)
    def analyze_trials(self):
        if self.dirty or not self.session.get('result'): raise ValueError('Compute traces for the current ROI set first.')
        if not self.session.get('sync'): raise ValueError('Import synchronization MAT before trial analysis.')
        cfg=self.config()
        def done(paths):
            window=HeatmapWindow(paths['png'],paths['fig'],self); window.show(); self.analysis_windows.append(window)
            window.finished.connect(lambda _=0,w=window:self.analysis_windows.remove(w) if w in self.analysis_windows else None)
            self.log.appendPlainText('Trial response analysis:\n'+'\n'.join(f'{key}: {value}' for key,value in paths.items()))
            self.banner.setText(f'Trial heatmap ready: {paths["cells"]} cells × {paths["trials"]} trials. ROIs sorted by mean response during stimulus.')
        self.run(partial(B.analyze_trial_responses,self.session,cfg),done)
    def closeEvent(self,event):
        if self.worker and self.worker.isRunning():
            Q.QMessageBox.information(self,'Processing','Wait for the current step to finish before closing.'); event.ignore(); return
        event.accept()

if __name__=='__main__':
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('TwoPhotonStudio.Desktop')
    app=Q.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(str(Path(__file__).resolve().parent/'assets/twophoton-studio.ico')))
    window=Studio(); window.show(); sys.exit(app.exec_())
