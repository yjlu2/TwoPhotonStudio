"""Read-only source paths with conventional copy actions."""
from PyQt5 import QtCore, QtGui, QtWidgets as Q


class CopyPaths:
    def copy_paths(self):
        raise NotImplementedError

    def setup_copy(self):
        self.copy_shortcut=Q.QShortcut(QtGui.QKeySequence.Copy,self)
        self.copy_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.copy_shortcut.activated.connect(self.copy_paths)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.copy_menu)

    def copy_menu(self,pos):
        menu=Q.QMenu(self)
        action=menu.addAction('Copy full path(s)'); action.setShortcut(QtGui.QKeySequence.Copy)
        action.triggered.connect(self.copy_paths)
        menu.exec_(self.viewport().mapToGlobal(pos))


class PathList(Q.QListWidget,CopyPaths):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setSelectionMode(Q.QAbstractItemView.ExtendedSelection)
        self.setup_copy()

    def copy_paths(self):
        items=sorted(self.selectedItems(),key=self.row)
        if not items and self.currentItem() is not None: items=[self.currentItem()]
        if items: Q.QApplication.clipboard().setText('\n'.join(i.text() for i in items))

    def copy_menu(self,pos):
        item=self.itemAt(pos)
        if item is not None and not item.isSelected():
            self.clearSelection(); item.setSelected(True); self.setCurrentItem(item)
        super().copy_menu(pos)


class PathTable(Q.QTableWidget,CopyPaths):
    def __init__(self,parent=None):
        super().__init__(0,3,parent)
        self.setSelectionBehavior(Q.QAbstractItemView.SelectRows)
        self.setSelectionMode(Q.QAbstractItemView.ExtendedSelection)
        self.setup_copy()

    def copy_paths(self):
        rows=sorted({i.row() for i in self.selectedIndexes()})
        if not rows and self.currentRow()>=0: rows=[self.currentRow()]
        paths=[self.item(row,2).text() for row in rows if self.item(row,2)]
        if paths: Q.QApplication.clipboard().setText('\n'.join(paths))

    def copy_menu(self,pos):
        item=self.itemAt(pos)
        if item is not None and not item.isSelected():
            self.clearSelection(); self.selectRow(item.row())
        super().copy_menu(pos)
