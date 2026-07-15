import wx
import wx.aui
import wx.lib.agw.hyperlink as hl
from EZannot import __version__


HOME_TAB_TITLE='Welcome'


def find_page_index(notebook,title):

	for i in range(notebook.GetPageCount()):
		if notebook.GetPageText(i)==title:
			return i
	return None


def open_or_select_page(notebook,title,panel_factory):

	idx=find_page_index(notebook,title)
	if idx is not None:
		notebook.SetSelection(idx)
		return notebook.GetPage(idx)
	panel=panel_factory()
	notebook.AddPage(panel,title,select=True)
	return panel


from .gui_training import PanelLv1_TrainingModule
from .gui_annotating import PanelLv1_AnnotationModule
from .gui_processing import PanelLv1_ProcessingModule



class InitialPanel(wx.Panel):

	def __init__(self,parent):

		super().__init__(parent)
		self.notebook=parent
		self.dispaly_window()


	def dispaly_window(self):

		panel=self
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		self.text_welcome=wx.StaticText(panel,label='Welcome to EZannot!',style=wx.ALIGN_CENTER|wx.ST_ELLIPSIZE_END)
		boxsizer.Add(0,60,0)
		boxsizer.Add(self.text_welcome,0,wx.LEFT|wx.RIGHT|wx.EXPAND,5)
		boxsizer.Add(0,60,0)
		self.text_developers=wx.StaticText(panel,label='\nCreated by Yujia Hu\n\n\n\nContributors: Bobby Tomlinson\n',style=wx.ALIGN_CENTER|wx.ST_ELLIPSIZE_END)
		boxsizer.Add(self.text_developers,0,wx.LEFT|wx.RIGHT|wx.EXPAND,5)
		boxsizer.Add(0,60,0)
		
		links=wx.BoxSizer(wx.HORIZONTAL)
		homepage=hl.HyperLinkCtrl(panel,0,'Home Page',URL='https://github.com/yujiahu415/EZannot')
		userguide=hl.HyperLinkCtrl(panel,0,'Extended Guide',URL='https://github.com/yujiahu415/EZannot/blob/master/EZannot_extended_user_guide.pdf')
		links.Add(homepage,0,wx.LEFT|wx.EXPAND,10)
		links.Add(userguide,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(links,0,wx.ALIGN_CENTER,50)
		boxsizer.Add(0,50,0)

		module_modules=wx.BoxSizer(wx.HORIZONTAL)
		button_train=wx.Button(panel,label='Training Module',size=(250,40))
		button_train.Bind(wx.EVT_BUTTON,self.panel_train)
		wx.Button.SetToolTip(button_train,'You can train and test an Annotator here. Annotators can automatically annotate images for you, which saves huge labor. Depending on the annotation precision, you may or may not need to do manual corrections.')
		button_annotate=wx.Button(panel,label='Annotation Module',size=(250,40))
		button_annotate.Bind(wx.EVT_BUTTON,self.panel_annotate)
		wx.Button.SetToolTip(button_annotate,'You can use a trained Annotator for automatic annotation. You can also perform AI-assisted semi-manual annotations to get a small set of initial training data for training an Annotator, or correct the annotations done by an Annotator.')
		button_process=wx.Button(panel,label='Process Module',size=(250,40))
		button_process.Bind(wx.EVT_BUTTON,self.panel_process)
		wx.Button.SetToolTip(button_process,'You can automatically measure all the annotated objects. You can also divide large annotated images into smaller tiles and preserve the annotations.')
		module_modules.Add(button_train,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_modules.Add(button_annotate,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_modules.Add(button_process,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_modules,0,wx.ALIGN_CENTER,50)
		boxsizer.Add(0,50,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def panel_train(self,event):

		open_or_select_page(self.notebook,'Training Module',lambda:PanelLv1_TrainingModule(self.notebook))


	def panel_annotate(self,event):

		open_or_select_page(self.notebook,'Annotation Module',lambda:PanelLv1_AnnotationModule(self.notebook))


	def panel_process(self,event):

		open_or_select_page(self.notebook,'Processing Module',lambda:PanelLv1_ProcessingModule(self.notebook))



class MainFrame(wx.Frame):

	def __init__(self):
		super().__init__(None,title='EZannot v'+str(__version__))
		self.SetSize((1000,500))

		self.aui_manager=wx.aui.AuiManager()
		self.aui_manager.SetManagedWindow(self)

		self.notebook=wx.aui.AuiNotebook(self)
		self.aui_manager.AddPane(self.notebook,wx.aui.AuiPaneInfo().CenterPane())
		self.notebook.Bind(wx.aui.EVT_AUINOTEBOOK_PAGE_CLOSE,self.on_page_close)

		panel=InitialPanel(self.notebook)
		self.notebook.AddPage(panel,HOME_TAB_TITLE,select=True)

		sizer=wx.BoxSizer(wx.VERTICAL)
		sizer.Add(self.notebook,1,wx.EXPAND)
		self.SetSizer(sizer)

		self.aui_manager.Update()
		self.Centre()
		self.Show()


	def on_page_close(self,event):

		if self.notebook.GetPageText(event.GetSelection())==HOME_TAB_TITLE:
			event.Veto()
			return
		event.Skip()



def main_window():

	app=wx.App()
	MainFrame()
	print('The user interface initialized!')
	app.MainLoop()


if __name__=='__main__':

	main_window()
