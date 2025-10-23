import os
import cv2
import wx
import wx.aui
import wx.lib.agw.hyperlink as hl
import json
import random
import shutil
import itertools
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.ndimage import rotate
from PIL import Image
from screeninfo import get_monitors
from EZannot.sam2.build_sam import build_sam2
from EZannot.sam2.sam2_image_predictor import SAM2ImagePredictor
from EZannot import __version__
from .gui_training import PanelLv1_TrainingModule
from .gui_annotating import PanelLv1_AnnotationModule
from .annotator import Annotator,AutoAnnotation
from .tools import read_annotation,measure_annotation



the_absolute_current_path=str(Path(__file__).resolve().parent)



class ColorPicker(wx.Dialog):

	def __init__(self,parent,title,name_and_color):

		super(ColorPicker,self).__init__(parent=None,title=title,size=(200,200))

		self.name_and_color=name_and_color
		name=self.name_and_color[0]
		hex_color=self.name_and_color[1].lstrip('#')
		color=tuple(int(hex_color[i:i+2],16) for i in (0,2,4))

		boxsizer=wx.BoxSizer(wx.VERTICAL)

		self.color_picker=wx.ColourPickerCtrl(self,colour=color)

		button=wx.Button(self,wx.ID_OK,label='Apply')

		boxsizer.Add(0,10,0)
		boxsizer.Add(self.color_picker,0,wx.ALL|wx.CENTER,10)
		boxsizer.Add(button,0,wx.ALL|wx.CENTER,10)
		boxsizer.Add(0,10,0)

		self.SetSizer(boxsizer)



class PanelLv1_ProcessingModule(wx.Panel):

	def __init__(self,parent):

		super().__init__(parent)
		self.notebook=parent
		self.dispaly_window()


	def dispaly_window(self):

		panel=self
		boxsizer=wx.BoxSizer(wx.VERTICAL)
		boxsizer.Add(0,60,0)

		button_measureannotations=wx.Button(panel,label='Measure Annotations',size=(300,40))
		button_measureannotations.Bind(wx.EVT_BUTTON,self.measure_annotations)
		wx.Button.SetToolTip(button_measureannotations,'Automatically provide quantitative measurements (such as center,area,height,width,perimeter,roundness,intensity) for each annotated object.')
		boxsizer.Add(button_measureannotations,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,5,0)

		button_tileannotations=wx.Button(panel,label='Tile Annotations',size=(300,40))
		button_tileannotations.Bind(wx.EVT_BUTTON,self.tile_annotations)
		wx.Button.SetToolTip(button_tileannotations,'Divide large annotated images into smaller tiles and preserve the annotations.')
		boxsizer.Add(button_tileannotations,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,50,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def measure_annotations(self,event):

		panel=PanelLv2_MeasureAnnotations(self.notebook)
		title='Measure Annotations'
		self.notebook.AddPage(panel,title,select=True)


	def tile_annotations(self,event):

		panel=PanelLv2_TileAnnotations(self.notebook)
		title='Tile Annotations'
		self.notebook.AddPage(panel,title,select=True)



class PanelLv2_MeasureAnnotations(wx.Panel):

	def __init__(self,parent):

		super().__init__(parent)
		self.notebook=parent
		self.path_to_images=None
		self.result_path=None
		self.color_map={}
		self.show_ids=False

		self.display_window()


	def display_window(self):

		panel=self
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_input=wx.BoxSizer(wx.HORIZONTAL)
		button_input=wx.Button(panel,label='Select the folder that stores\nannotated images for measurements',size=(300,40))
		button_input.Bind(wx.EVT_BUTTON,self.select_images)
		wx.Button.SetToolTip(button_input,'Select the folder that stores all the annotated images. You also need to put the annotation file(s) in the same folder, and EZannot will decode the annotations in the annotation file(s) automatically.')
		self.text_input=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_input.Add(button_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_input.Add(self.text_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_outputfolder=wx.BoxSizer(wx.HORIZONTAL)
		button_outputfolder=wx.Button(panel,label='Select a folder to store\nthe measurements',size=(300,40))
		button_outputfolder.Bind(wx.EVT_BUTTON,self.select_outpath)
		wx.Button.SetToolTip(button_outputfolder,'Copies of annotated images and the quantitative measurements will be stored in this folder.')
		self.text_outputfolder=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_outputfolder.Add(button_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_outputfolder.Add(self.text_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_classes=wx.BoxSizer(wx.HORIZONTAL)
		button_classes=wx.Button(panel,label='Specify the colors for\nthe annotated objects',size=(300,40))
		button_classes.Bind(wx.EVT_BUTTON,self.specify_colors)
		wx.Button.SetToolTip(button_classes,'Specify the colors that represent the annotated objects and EZannot will output the annotated images.')
		self.text_classes=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_classes.Add(button_classes,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_classes.Add(self.text_classes,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_classes,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_measureannotation=wx.Button(panel,label='Measure the annotated',size=(300,40))
		button_measureannotation.Bind(wx.EVT_BUTTON,self.measure_annotations)
		wx.Button.SetToolTip(button_measureannotation,'Calculate diverse quantitative measurements for each annotated object.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_measureannotation,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_images(self,event):

		wildcard='Image files(*.jpg;*.jpeg;*.png;*.tif;*.tiff)|*.jpg;*.jpeg;*.png;*.tif;*.tiff'
		dialog=wx.FileDialog(self,'Select images(s)','','',wildcard,style=wx.FD_MULTIPLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_images=dialog.GetPaths()
			self.path_to_images.sort()
			path=os.path.dirname(self.path_to_images[0])
			self.text_input.SetLabel('Select: '+str(len(self.path_to_images))+' images in'+str(path)+'.')
		dialog.Destroy()


	def select_outpath(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.result_path=dialog.GetPath()
			self.text_outputfolder.SetLabel('The annotated images will be in: '+self.result_path+'.')
		dialog.Destroy()


	def specify_colors(self,event):

		if self.path_to_images is None:

			wx.MessageBox('No input images(s).','Error',wx.OK|wx.ICON_ERROR)

		else:

			annotation_files=[]
			color_map={}
			self.color_map={}
			classnames=[]
			entry=None
			for i in os.listdir(os.path.dirname(self.path_to_images[0])):
				if i.endswith('.json'):
					annotation_files.append(os.path.join(os.path.dirname(self.path_to_images[0]),i))

			if len(annotation_files)>0:
				for annotation_file in annotation_files:
					if os.path.exists(annotation_file):
						annotation=json.load(open(annotation_file))
						for i in annotation['categories']:
							if i['id']>0:
								classname=i['name']
								if classname not in classnames:
									classnames.append(classname)

			for i in classnames:
				color_map[i]='#%02x%02x%02x'%(random.randint(0,255),random.randint(0,255),random.randint(0,255))

			if len(color_map)>0:
				for classname in color_map:
					dialog=ColorPicker(self,f'{classname}',[classname,color_map[classname]])
					if dialog.ShowModal()==wx.ID_OK:
						(r,b,g,_)=dialog.color_picker.GetColour()
						self.color_map[classname]=(r,b,g)
					dialog.Destroy()
				self.text_classes.SetLabel('Classname:color: '+str(self.color_map)+'.')
			else:
				self.text_classes.SetLabel('None.')

			dialog=wx.MessageDialog(self,'Show the IDs for\nall the annotated objects?','Show IDs?',wx.YES_NO|wx.ICON_QUESTION)
			if dialog.ShowModal()==wx.ID_YES:
				self.show_ids=True
			else:
				self.show_ids=False
			dialog.Destroy()


	def measure_annotations(self,event):

		if self.path_to_images is None or self.result_path is None:

			wx.MessageBox('No input images(s) / output folder.','Error',wx.OK|wx.ICON_ERROR)

		else:

			information=read_annotation(os.path.dirname(self.image_paths[0]))

			measure_annotation(path_to_images,result_path,information,color_map,show_ids=False,threshold=None)

			annotation_files=[]
			color_map={}
			self.color_map={}
			classnames=''
			entry=None
			for i in os.listdir(os.path.dirname(self.path_to_images[0])):
				if i.endswith('.json'):
					annotation_files.append(os.path.join(os.path.dirname(self.path_to_images[0]),i))

			if len(annotation_files)>0:
				for annotation_file in annotation_files:
					if os.path.exists(annotation_file):
						annotation=json.load(open(annotation_file))
						for i in annotation['categories']:
							if i['id']>0:
								classnames=classnames+i['name']+','
				classnames=classnames[:-1]
				dialog=wx.MessageDialog(self,'Current classnames are: '+classnames+'.\nDo you want to modify the classnames?','Modify classnames?',wx.YES_NO|wx.ICON_QUESTION)
				if dialog.ShowModal()==wx.ID_YES:
					dialog1=wx.TextEntryDialog(self,'Enter the names of objects to annotate\n(use "," to separate each name)','Object class names',value=classnames)
					if dialog1.ShowModal()==wx.ID_OK:
						entry=dialog1.GetValue()
					dialog1.Destroy()
				else:
					entry=classnames
				dialog.Destroy()
			else:
				dialog=wx.TextEntryDialog(self,'Enter the names of objects to annotate\n(use "," to separate each name)','Object class names')
				if dialog.ShowModal()==wx.ID_OK:
					entry=dialog.GetValue()
				dialog.Destroy()

			if entry:
				try:
					for i in entry.split(','):
						color_map[i]='#%02x%02x%02x'%(random.randint(0,255),random.randint(0,255),random.randint(0,255))
				except:
					color_map={}
					wx.MessageBox('Please enter the object class names in\ncorrect format! For example: apple,orange,pear','Error',wx.OK|wx.ICON_ERROR)

			if len(color_map)>0:
				for classname in color_map:
					dialog=ColorPicker(self,f'{classname}',[classname,color_map[classname]])
					if dialog.ShowModal()==wx.ID_OK:
						(r,b,g,_)=dialog.color_picker.GetColour()
						self.color_map[classname]=(r,b,g)
					dialog.Destroy()
				self.text_classes.SetLabel('Classname:color: '+str(self.color_map)+'.')
			else:
				self.text_classes.SetLabel('None.')

			dialog=wx.MessageDialog(self,'Show the IDs for all the annotated\nwhen exporting the measurements?','Show IDs?',wx.YES_NO|wx.ICON_QUESTION)
			if dialog.ShowModal()==wx.ID_YES:
				self.show_ids=True
			else:
				self.show_ids=False
			dialog.Destroy()


class WindowLv3_AnnotateImages(wx.Frame):

	def __init__(self,parent,title,path_to_images,result_path,color_map,aug_methods,model_cp=None,model_cfg=None,show_ids=False):

		monitor=get_monitors()[0]
		monitor_w,monitor_h=monitor.width,monitor.height

		super().__init__(parent,title=title,pos=(10,0),size=(get_monitors()[0].width-20,get_monitors()[0].height-50))

		self.image_paths=path_to_images
		self.result_path=result_path
		self.color_map=color_map
		self.aug_methods=aug_methods
		self.model_cp=model_cp
		self.model_cfg=model_cfg
		self.show_ids=show_ids
		self.current_image_id=0
		self.current_image=None
		self.current_segmentation=None
		self.current_polygon=[]
		self.current_classname=list(self.color_map.keys())[0]
		self.information=read_annotation(os.path.dirname(self.image_paths[0]),self.color_map)
		self.foreground_points=[]
		self.background_points=[]
		self.selected_point=None
		self.start_modify=False
		self.show_name=False
		self.AI_help=False
		self.scale=1.0
		self.min_scale=0.25
		self.max_scale=8.0
		self.zoom_step=1.25

		self.init_ui()
		self.load_current_image()


	def sam2_model(self):

		device='cuda' if torch.cuda.is_available() else 'cpu'
		predictor=SAM2ImagePredictor(build_sam2(self.model_cfg,self.model_cp,device=device))
		return predictor


	def init_ui(self):

		panel=wx.Panel(self)
		vbox=wx.BoxSizer(wx.VERTICAL)
		hbox=wx.BoxSizer(wx.HORIZONTAL)

		self.ai_button=wx.ToggleButton(panel,label='AI Help: OFF',size=(150,30))
		self.ai_button.Bind(wx.EVT_TOGGLEBUTTON,self.toggle_ai)
		hbox.Add(self.ai_button,flag=wx.ALL,border=2)

		self.prev_button=wx.Button(panel,label='← Prev',size=(150,30))
		self.prev_button.Bind(wx.EVT_BUTTON,self.previous_image)
		hbox.Add(self.prev_button,flag=wx.ALL,border=2)

		self.next_button=wx.Button(panel,label='Next →',size=(150,30))
		self.next_button.Bind(wx.EVT_BUTTON,self.next_image)
		hbox.Add(self.next_button,flag=wx.ALL,border=2)

		self.delete_button=wx.Button(panel,label='Delete',size=(150,30))
		self.delete_button.Bind(wx.EVT_BUTTON,self.delete_image)
		hbox.Add(self.delete_button,flag=wx.ALL,border=2)

		self.export_button=wx.Button(panel,label='Export Annotations',size=(150,30))
		self.export_button.Bind(wx.EVT_BUTTON,self.export_annotations)
		hbox.Add(self.export_button,flag=wx.ALL,border=2)

		self.measure_button=wx.Button(panel,label='Measure Annotations',size=(150,30))
		self.measure_button.Bind(wx.EVT_BUTTON,self.measure_annotations)
		hbox.Add(self.measure_button,flag=wx.ALL,border=2)
		vbox.Add(hbox,flag=wx.ALIGN_CENTER|wx.TOP,border=5)

		self.scrolled_canvas=wx.ScrolledWindow(panel,style=wx.VSCROLL|wx.HSCROLL)
		self.scrolled_canvas.SetScrollRate(10,10)
		self.canvas=wx.Panel(self.scrolled_canvas,pos=(10,0),size=(get_monitors()[0].width-20,get_monitors()[0].height-50))
		self.scrolled_canvas.SetBackgroundColour('black')

		self.canvas.Bind(wx.EVT_PAINT,self.on_paint)
		self.canvas.Bind(wx.EVT_LEFT_DOWN,self.on_left_click)
		self.canvas.Bind(wx.EVT_RIGHT_DOWN,self.on_right_click)
		self.canvas.Bind(wx.EVT_MOTION,self.on_left_move)
		self.canvas.Bind(wx.EVT_LEFT_UP,self.on_left_up)
		self.canvas.Bind(wx.EVT_MOUSEWHEEL,self.on_mousewheel)

		self.scrolled_canvas.SetSizer(wx.BoxSizer(wx.VERTICAL))
		self.scrolled_canvas.GetSizer().Add(self.canvas,proportion=1,flag=wx.EXPAND|wx.ALL,border=5)
		vbox.Add(self.scrolled_canvas,proportion=1,flag=wx.EXPAND|wx.ALL,border=5)

		panel.SetSizer(vbox)
		self.Bind(wx.EVT_CHAR_HOOK,self.on_key_press)
		self.Show()


	def toggle_ai(self,event):

		if self.model_cp is None or self.model_cfg is None:

			self.ai_button.SetLabel('AI Help: OFF')
			wx.MessageBox('SAM2 model has not been set up.','Error',wx.ICON_ERROR)

		else:

			self.AI_help=self.ai_button.GetValue()
			self.ai_button.SetLabel(f'AI Help: {"ON" if self.AI_help else "OFF"}')

			if self.AI_help:
				image=Image.open(self.image_paths[self.current_image_id])
				image=np.array(image.convert('RGB'))
				self.sam2=self.sam2_model()
				self.sam2.set_image(image)

		self.canvas.SetFocus()


	def load_current_image(self):

		if self.image_paths:
			path=self.image_paths[self.current_image_id]
			self.current_image=wx.Image(path,wx.BITMAP_TYPE_ANY)
			img_width,img_height=self.current_image.GetSize()
			self.scrolled_canvas.SetVirtualSize((img_width,img_height))
			self.canvas.SetSize((img_width,img_height))
			self.scrolled_canvas.Scroll(0,0)
			image_name=os.path.basename(path)
			if image_name not in self.information:
				self.information[image_name]={'polygons':[],'class_names':[]}
			self.current_polygon=[]
			self.foreground_points=[]
			self.background_points=[]
			self.scale=1.0
			self.canvas.Refresh()

			if self.AI_help:
				image=Image.open(path)
				image=np.array(image.convert('RGB'))
				self.sam2=self.sam2_model()
				self.sam2.set_image(image)


	def previous_image(self,event):

		if self.image_paths and self.current_image_id>0:
			self.current_image_id-=1
			self.load_current_image()
		self.canvas.SetFocus()


	def next_image(self,event):

		if self.image_paths and self.current_image_id<len(self.image_paths)-1:
			self.current_image_id+=1
			self.load_current_image()
		self.canvas.SetFocus()


	def delete_image(self,event):

		if self.image_paths:
			path=self.image_paths[self.current_image_id]
			self.image_paths.remove(path)
			image_name=os.path.basename(path)
			del self.information[image_name]
			self.load_current_image()
		self.canvas.SetFocus()


	def on_paint(self,event):

		if self.current_image is None:
			return

		dc=wx.PaintDC(self.canvas)
		w,h=self.current_image.GetSize()
		scaled_image=self.current_image.Scale(int(w*self.scale),int(h*self.scale),wx.IMAGE_QUALITY_HIGH)
		dc.DrawBitmap(wx.Bitmap(scaled_image),0,0,True)
		image_name=os.path.basename(self.image_paths[self.current_image_id])
		polygons=self.information[image_name]['polygons']
		class_names=self.information[image_name]['class_names']

		if len(polygons)>0:
			for i,polygon in enumerate(polygons):
				color=self.color_map[class_names[i]]
				pen=wx.Pen(wx.Colour(*color),width=2)
				dc.SetPen(pen)
				dc.DrawLines([(int(x*self.scale),int(y*self.scale)) for x,y in polygon])
				if self.start_modify:
					brush=wx.Brush(wx.Colour(*color))
					dc.SetBrush(brush)
					for x,y in polygon:
						dc.DrawCircle(int(x*self.scale),int(y*self.scale),4)
				if self.show_name:
					x_max=int(max(x for x,y in polygon)*self.scale)
					x_min=int(min(x for x,y in polygon)*self.scale)
					y_max=int(max(y for x,y in polygon)*self.scale)
					y_min=int(min(y for x,y in polygon)*self.scale)
					cx=int((x_max+x_min)/2)
					cy=int((y_max+y_min)/2)
					dc.SetTextForeground(wx.Colour(*color))
					dc.SetFont(wx.Font(wx.FontInfo(15).FaceName('Arial')))
					dc.DrawText(str(class_names[i]),cx,cy)

		if len(self.current_polygon)>0:
			current_polygon=[i for i in self.current_polygon]
			current_polygon.append(current_polygon[0])
			color=self.color_map[self.current_classname]
			brush=wx.Brush(wx.Colour(*color))
			dc.SetBrush(brush)
			for x,y in current_polygon:
				dc.DrawCircle(int(x*self.scale),int(y*self.scale),4)
			pen=wx.Pen(wx.Colour(*color),width=2)
			dc.SetPen(pen)
			dc.DrawLines([(int(x*self.scale),int(y*self.scale)) for x,y in current_polygon])


	def on_left_click(self,event):

		x,y=event.GetX(),event.GetY()

		if self.start_modify:

			image_name=os.path.basename(self.image_paths[self.current_image_id])
			for i,polygon in enumerate(self.information[image_name]['polygons']):
				for j,(px,py) in enumerate(polygon):
					if abs(px-int(x/self.scale))<5 and abs(py-int(y/self.scale))<5:
						self.selected_point=(polygon,j,i)
						return

		else:

			if self.AI_help:
				self.foreground_points.append([int(x/self.scale),int(y/self.scale)])
				points=self.foreground_points+self.background_points
				labels=[1 for i in range(len(self.foreground_points))]+[0 for i in range(len(self.background_points))]
				masks,scores,logits=self.sam2.predict(point_coords=np.array(points),point_labels=np.array(labels))
				mask=masks[np.argsort(scores)[::-1]][0]
				self.current_polygon=self.mask_to_polygon(mask)
			else:
				self.current_polygon.append((int(x/self.scale),int(y/self.scale)))

		self.canvas.Refresh()


	def on_right_click(self,event):

		x,y=event.GetX(),event.GetY()

		if self.start_modify:

			return

		else:

			if len(self.current_polygon)>0:

				if self.AI_help:
					self.background_points.append([int(x/self.scale),int(y/self.scale)])
					points=self.foreground_points+self.background_points
					labels=[1 for i in range(len(self.foreground_points))]+[0 for i in range(len(self.background_points))]
					masks,scores,logits=self.sam2.predict(point_coords=np.array(points),point_labels=np.array(labels))
					mask=masks[np.argsort(scores)[::-1]][0]
					self.current_polygon=self.mask_to_polygon(mask)
				else:
					self.current_polygon.pop()

			else:

				to_delete=[]
				image_name=os.path.basename(self.image_paths[self.current_image_id])
				polygons=self.information[image_name]['polygons']
				class_names=self.information[image_name]['class_names']
				if len(polygons)>0:
					for i,polygon in enumerate(polygons):
						x_max=max(x for x,y in polygon)
						x_min=min(x for x,y in polygon)
						y_max=max(y for x,y in polygon)
						y_min=min(y for x,y in polygon)
						if x_min<=int(x/self.scale)<=x_max and y_min<=int(y/self.scale)<=y_max:
							to_delete.append(i)
				if len(to_delete)>0:
					for i in sorted(to_delete,reverse=True):
						del self.information[image_name]['polygons'][i]
						del self.information[image_name]['class_names'][i]

		self.canvas.Refresh()


	def on_key_press(self,event):

		key_code=event.GetKeyCode()

		if event.GetKeyCode()==wx.WXK_RETURN:
			if len(self.current_polygon)>2:
				classnames=sorted(list(self.color_map.keys()))
				current_index=classnames.index(self.current_classname)
				dialog=wx.SingleChoiceDialog(self,message='Choose object class name',caption='Class Name',choices=classnames)
				dialog.SetSelection(current_index)
				if dialog.ShowModal()==wx.ID_OK:
					self.current_classname=dialog.GetStringSelection()
					if len(self.current_polygon)>0:
						self.current_polygon.append(self.current_polygon[0])
						image_name=os.path.basename(self.image_paths[self.current_image_id])
						self.information[image_name]['polygons'].append(self.current_polygon)
						self.information[image_name]['class_names'].append(self.current_classname)
				dialog.Destroy()
				self.current_polygon=[]
				self.foreground_points=[]
				self.background_points=[]
				self.canvas.Refresh()
		elif key_code==wx.WXK_LEFT:
			self.previous_image(None)
		elif key_code==wx.WXK_RIGHT:
			self.next_image(None)
		elif event.GetKeyCode()==wx.WXK_SHIFT:
			if self.start_modify:
				self.start_modify=False
			else:
				self.start_modify=True
			self.canvas.Refresh()
		elif event.GetKeyCode()==wx.WXK_SPACE:
			if self.show_name:
				self.show_name=False
			else:
				self.show_name=True
			self.canvas.Refresh()
		elif event.GetKeyCode()==wx.WXK_ESCAPE:
			self.current_polygon=[]
			self.foreground_points=[]
			self.background_points=[]
			self.canvas.Refresh()
		else:
			event.Skip()


	def on_left_move(self,event):

		if self.selected_point is not None and event.Dragging() and event.LeftIsDown():
			polygon,j,i=self.selected_point
			x,y=event.GetX(),event.GetY()
			polygon[j]=(int(x/self.scale),int(y/self.scale))
			image_name=os.path.basename(self.image_paths[self.current_image_id])
			self.information[image_name]['polygons'][i]=polygon
			self.canvas.Refresh()


	def on_left_up(self,event):

		self.selected_point=None


	def on_mousewheel(self,event):

		if self.current_image is None:
			return

		if self.start_modify:
			return

		rotation=event.GetWheelRotation()
		if rotation>0:
			self.scale=min(self.scale*self.zoom_step,self.max_scale)
		else:
			self.scale=max(self.scale/self.zoom_step,self.min_scale)

		new_w=int(self.current_image.GetWidth()*self.scale)
		new_h=int(self.current_image.GetHeight()*self.scale)
		self.scrolled_canvas.SetVirtualSize((new_w,new_h))
		self.canvas.Refresh()


	def export_annotations(self,event):

		if not self.information:
			wx.MessageBox('No annotations to export.','Error',wx.ICON_ERROR)
			return

		generate_annotation(os.path.dirname(self.image_paths[0]),self.information,self.result_path,self.result_path,self.aug_methods,self.color_map)
		generate_annotation(os.path.dirname(self.image_paths[0]),self.information,os.path.dirname(self.image_paths[0]),self.result_path,[],self.color_map)

		wx.MessageBox('Annotations exported successfully.','Success',wx.ICON_INFORMATION)

		self.canvas.SetFocus()


	def measure_annotations(self,event):

		if not self.information:
			wx.MessageBox('No annotations to measure.','Error',wx.ICON_ERROR)
			return

		measure_annotation(os.path.dirname(self.image_paths[0]),self.out_path,self.information,self.color_map,show_ids=self.show_ids,threshold=None)


		self.canvas.SetFocus()



class PanelLv2_TileAnnotations(wx.Panel):

	def __init__(self,parent):

		super().__init__(parent)
		self.notebook=parent
		self.path_to_images=None
		self.path_to_annotator=None
		self.object_kinds=None
		self.detection_threshold={}
		self.filters={}

		self.display_window()


	def display_window(self):

		panel=self
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_input=wx.BoxSizer(wx.HORIZONTAL)
		button_input=wx.Button(panel,label='Select the image(s)\nfor annotation',size=(300,40))
		button_input.Bind(wx.EVT_BUTTON,self.select_images)
		wx.Button.SetToolTip(button_input,'Select one or more images. Common image formats (jpg, png, tif) are supported. An annotation file will be generated in this folder')
		self.text_input=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_input.Add(button_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_input.Add(self.text_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_model=wx.BoxSizer(wx.HORIZONTAL)
		button_model=wx.Button(panel,label='Select a trained Annotator\nfor automatic annotation',size=(300,40))
		button_model.Bind(wx.EVT_BUTTON,self.select_model)
		wx.Button.SetToolTip(button_model,'A trained Annotator can annotate the objects of your interest in images.')
		self.text_model=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_model.Add(button_model,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_model.Add(self.text_model,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_model,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_filters=wx.BoxSizer(wx.HORIZONTAL)
		button_filters=wx.Button(panel,label='Specify the filters to\nexclude unwanted annotations',size=(300,40))
		button_filters.Bind(wx.EVT_BUTTON,self.specify_filters)
		wx.Button.SetToolTip(button_filters,'Select filters such as area, perimeter, roundness (1 is circle, higer value means less round), height, and width, and specify the minimum and maximum values of these filters.')
		self.text_filters=wx.StaticText(panel,label='Default: None',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_filters.Add(button_filters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_filters.Add(self.text_filters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_filters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_startannotation=wx.Button(panel,label='Start to annotate images',size=(300,40))
		button_startannotation.Bind(wx.EVT_BUTTON,self.start_annotation)
		wx.Button.SetToolTip(button_startannotation,'Automatically annotate objects in images.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_startannotation,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_images(self,event):

		wildcard='Image files(*.jpg;*.jpeg;*.png;*.tif;*.tiff)|*.jpg;*.jpeg;*.png;*.tif;*.tiff'
		dialog=wx.FileDialog(self,'Select images(s)','','',wildcard,style=wx.FD_MULTIPLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_images=dialog.GetPaths()
			self.path_to_images.sort()
			path=os.path.dirname(self.path_to_images[0])
			self.text_input.SetLabel('Select: '+str(len(self.path_to_images))+' images in'+str(path)+'.')
		dialog.Destroy()


	def select_model(self,event):

		annotator_path=os.path.join(the_absolute_current_path,'annotators')
		annotators=[i for i in os.listdir(annotator_path) if os.path.isdir(os.path.join(annotator_path,i))]
		if '__pycache__' in annotators:
			annotators.remove('__pycache__')
		if '__init__' in annotators:
			annotators.remove('__init__')
		if '__init__.py' in annotators:
			annotators.remove('__init__.py')
		annotators.sort()
		if 'Choose a new directory of the Annotator' not in annotators:
			annotators.append('Choose a new directory of the Annotator')

		dialog=wx.SingleChoiceDialog(self,message='Select an Annotator for automatic annotation.',caption='Select an Annotator',choices=annotators)
		if dialog.ShowModal()==wx.ID_OK:
			annotator=dialog.GetStringSelection()
			if annotator=='Choose a new directory of the Annotator':
				dialog1=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
				if dialog1.ShowModal()==wx.ID_OK:
					self.path_to_annotator=dialog1.GetPath()
				else:
					self.path_to_annotator=None
				dialog1.Destroy()
			else:
				self.path_to_annotator=os.path.join(annotator_path,annotator)
			with open(os.path.join(self.path_to_annotator,'model_parameters.txt')) as f:
				model_parameters=f.read()
			object_names=json.loads(model_parameters)['object_names']
			if len(object_names)>1:
				dialog1=wx.MultiChoiceDialog(self,message='Specify which obejct to annotate',
					caption='Object kind',choices=object_names)
				if dialog1.ShowModal()==wx.ID_OK:
					self.object_kinds=[object_names[i] for i in dialog1.GetSelections()]
				else:
					self.object_kinds=object_names
				dialog1.Destroy()
			else:
				self.object_kinds=object_names
			for object_name in self.object_kinds:
				dialog1=wx.NumberEntryDialog(self,'Detection threshold for '+str(object_name),'Enter an number between 0 and 100','Detection threshold for '+str(object_name),0,0,100)
				if dialog1.ShowModal()==wx.ID_OK:
					self.detection_threshold[object_name]=int(dialog1.GetValue())/100
				else:
					self.detection_threshold[object_name]=0
				dialog1.Destroy()
			self.text_model.SetLabel('Annotator: '+annotator+'; '+'The object kinds / detection threshold: '+str(self.detection_threshold)+'.')
		dialog.Destroy()

		if self.path_to_annotator is None:
			wx.MessageBox('No Annotator is selected.','No Annotator',wx.ICON_INFORMATION)
			self.text_model.SetLabel('No Annotator is selected.')


	def specify_filters(self,event):

		filters_choices=['area','perimeter','roundness','height','width']

		dialog=wx.MultiChoiceDialog(self,message='Select filters to exclude unwanted annotations',caption='Filters',choices=filters_choices)
		if dialog.ShowModal()==wx.ID_OK:
			selected_filters=[filters_choices[i] for i in dialog.GetSelections()]
		else:
			selected_filters=[]
		dialog.Destroy()

		for ft in selected_filters:
			dialog=wx.NumberEntryDialog(self,'The min value for '+str(ft),'The unit is pixel (except for roundness)','The min value for '+str(ft),0,0,100000000000000)
			values=[0,np.inf]
			if dialog.ShowModal()==wx.ID_OK:
				values[0]=int(dialog.GetValue())
			dialog.Destroy()
			dialog=wx.NumberEntryDialog(self,'The max value (enter 0 for infinity) for '+str(ft),'The unit is pixel (except for roundness)','The max value for '+str(ft),0,0,100000000000000)
			if dialog.ShowModal()==wx.ID_OK:
				value=int(dialog.GetValue())
				if value>0:
					values[1]=value
			dialog.Destroy()
			self.filters[ft]=values

		if len(self.filters)>0:
			self.text_filters.SetLabel('Filters: '+str(self.filters))
		else:
			self.text_filters.SetLabel('NO filters selected.')


	def start_annotation(self,event):

		if self.path_to_images is None or self.path_to_annotator is None:

			wx.MessageBox('No input images(s) / trained Annotator selected.','Error',wx.OK|wx.ICON_ERROR)

		else:
			
			AA=AutoAnnotation(self.path_to_images,self.path_to_annotator,self.object_kinds,detection_threshold=self.detection_threshold,filters=self.filters)
			AA.annotate_images()



class MainFrame(wx.Frame):

	def __init__(self):
		super().__init__(None,title=f'EZannot v{__version__}')
		self.SetSize((1000,500))

		self.aui_manager=wx.aui.AuiManager()
		self.aui_manager.SetManagedWindow(self)

		self.notebook=wx.aui.AuiNotebook(self)
		self.aui_manager.AddPane(self.notebook,wx.aui.AuiPaneInfo().CenterPane())

		panel=InitialPanel(self.notebook)
		title='Welcome'
		self.notebook.AddPage(panel,title,select=True)

		sizer=wx.BoxSizer(wx.VERTICAL)
		sizer.Add(self.notebook,1,wx.EXPAND)
		self.SetSizer(sizer)

		self.aui_manager.Update()
		self.Centre()
		self.Show()



def main_window():

	app=wx.App()
	MainFrame()
	print('The user interface initialized!')
	app.MainLoop()


if __name__=='__main__':

	main_window()

