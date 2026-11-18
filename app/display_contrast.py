"""Display-only linear brightness/contrast controls. Never modifies image pixels."""
import numpy as np
from PyQt5 import QtCore, QtWidgets as Q
import pyqtgraph as pg


class ContrastDialog(Q.QDialog):
    levels_changed=QtCore.pyqtSignal(float,float)
    auto_requested=QtCore.pyqtSignal()
    reset_requested=QtCore.pyqtSignal()

    def __init__(self,parent):
        super().__init__(parent)
        self.setWindowTitle('Brightness / Contrast')
        self.setWindowFlags(self.windowFlags()|QtCore.Qt.Tool)
        self.resize(450,490)
        self.updating=False; self.bounds=(0.,1.); self.levels=(0.,1.)
        layout=Q.QVBoxLayout(self)
        note=Q.QLabel('Display only: linear intensity mapping.\nROI detection, saved movie and fluorescence traces use original data.')
        note.setWordWrap(True); layout.addWidget(note)
        self.hist=pg.PlotWidget(); self.hist.setMinimumHeight(140)
        self.hist.setLabel('bottom','Intensity'); self.hist.setLabel('left','Pixels')
        self.curve=self.hist.plot(pen='#b8d9ef',fillLevel=0,brush=(70,130,170,70))
        self.region=pg.LinearRegionItem(values=(0,1),brush=(255,205,0,25))
        self.hist.addItem(self.region); self.region.sigRegionChangeFinished.connect(self.region_changed)
        layout.addWidget(self.hist,1)
        form=Q.QFormLayout(); layout.addLayout(form)
        self.spins=[]; self.sliders=[]
        for label in ('Minimum','Maximum'):
            row=Q.QHBoxLayout(); spin=Q.QDoubleSpinBox()
            spin.setDecimals(6); spin.setRange(-1e30,1e30)
            spin.setKeyboardTracking(False); spin.setMinimumWidth(135)
            slider=Q.QSlider(QtCore.Qt.Horizontal); slider.setRange(-1000,2000)
            row.addWidget(spin); row.addWidget(slider,1); form.addRow(label,row)
            self.spins.append(spin); self.sliders.append(slider)
        for i,spin in enumerate(self.spins):
            spin.valueChanged.connect(lambda _,i=i:self.spin_changed(i))
        for i,slider in enumerate(self.sliders):
            slider.valueChanged.connect(lambda value,i=i:self.limit_changed(i,value))
        self.brightness=Q.QSlider(QtCore.Qt.Horizontal); self.brightness.setRange(-1000,2000)
        self.contrast=Q.QSlider(QtCore.Qt.Horizontal); self.contrast.setRange(0,1000)
        form.addRow('Brightness',self.brightness); form.addRow('Contrast',self.contrast)
        self.brightness.valueChanged.connect(self.brightness_changed)
        self.contrast.valueChanged.connect(self.contrast_changed)
        actions=Q.QHBoxLayout(); layout.addLayout(actions)
        for label,signal in [('Auto',self.auto_requested),('Reset',self.reset_requested)]:
            button=Q.QPushButton(label); button.setMinimumHeight(34)
            button.clicked.connect(lambda _,s=signal:s.emit()); actions.addWidget(button)
        close=Q.QPushButton('Close'); close.setMinimumHeight(34)
        close.clicked.connect(self.close); actions.addWidget(close)

    def update_image(self,image,levels):
        a=np.asarray(image).reshape(-1)
        if a.size>500000: a=a[::int(np.ceil(a.size/500000))]
        a=a[np.isfinite(a)]
        if not a.size: return
        lo=float(a.min()); hi=float(a.max())
        if hi<=lo: hi=lo+max(1.,abs(lo)*1e-6)
        self.bounds=(lo,hi)
        counts,edges=np.histogram(a,bins=256,range=(lo,hi))
        self.curve.setData((edges[:-1]+edges[1:])/2,counts)
        self.sync(levels)

    def sync(self,levels):
        self.updating=True
        try:
            lo,hi=map(float,levels); self.levels=(lo,hi)
            a,b=self.bounds; span=b-a
            for spin,value in zip(self.spins,(lo,hi)): spin.setValue(value)
            for slider,value in zip(self.sliders,(lo,hi)): slider.setValue(round((value-a)/span*1000))
            self.brightness.setValue(round((b-(lo+hi)/2)/span*1000))
            self.contrast.setValue(round(500-125*np.log2(max(hi-lo,1e-30)/span)))
            self.region.setRegion((lo,hi))
        finally: self.updating=False

    def apply(self,lo,hi):
        if self.updating: return
        a,b=self.bounds; epsilon=max((b-a)*1e-8,1e-7)
        if hi<=lo: hi=lo+epsilon
        self.sync((lo,hi)); self.levels_changed.emit(float(lo),float(hi))

    def spin_changed(self,index):
        if self.updating: return
        lo,hi=[s.value() for s in self.spins]
        if lo>=hi and index==1: lo=hi-max((self.bounds[1]-self.bounds[0])*1e-6,1e-6)
        self.apply(lo,hi)

    def limit_changed(self,index,value):
        if self.updating: return
        lo,hi=self.levels; a,b=self.bounds; limit=a+value/1000*(b-a)
        if index==0: lo=limit
        else: hi=limit
        self.apply(lo,hi)

    def brightness_changed(self,value):
        if self.updating: return
        a,b=self.bounds; center=b-value/1000*(b-a); half=(self.levels[1]-self.levels[0])/2
        self.apply(center-half,center+half)

    def contrast_changed(self,value):
        if self.updating: return
        a,b=self.bounds; center=sum(self.levels)/2
        half=(b-a)*2**((500-value)/125)/2
        self.apply(center-half,center+half)

    def region_changed(self):
        if not self.updating: self.apply(*self.region.getRegion())
