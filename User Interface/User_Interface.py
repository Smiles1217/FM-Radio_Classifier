from tkinter import *
import tkinter.ttk as ttk
import threading
import Radio_Listener
from rtlsdr import RtlSdr
from functools import partial
# ==============================================
# Globals
# ==============================================

global station_read_event; station_read_event = threading.Event()
global station_collect_event; station_collect_event = threading.Event()

# ==============================================
# Constants
# ==============================================

# Genre List For Button Abstraction
Genres = ['Genre0', 'Genre1', 'Genre2', 'Genre3', 'Genre4', 'Genre5', 'Genre6', 'Genre7']

# UI Fonts
SCL_FONT = ('Helvetica Bold', 10)
BTN_FONT = ('Helvetica Bold', 12)
LBL_FONT = ('Helvetica Bold', 20)
TTL_FONT = ('Helvetica Bold', 36)

# ==============================================
# Global Event Triggers
# ==============================================

def Find_Genre(genre):
	# TODO: Define behavior based on genre - On repeat presses, increment through all stations listed as genre
	global window
	print(genre)
	pass

# ==============================================
# Window Definition
# ==============================================

class Button_Frame(Frame):
	def __init__(self, master=None, cnf={}, **kw):
		# Init Frame base class
		super().__init__(master=master, cnf=cnf, **kw)

		# Genre Selection Buttons
		self.btn_Genres = []
		for genre in Genres:
			self.btn_Genres.append(Button(self, text=genre, font=BTN_FONT, command=partial(Find_Genre, genre)))

		# Place Buttons in a 2-Wide Grid
		for i in range(0, len(self.btn_Genres)):
			self.btn_Genres[i].grid(column=int(i % 2), row=int(i / 2))
			pass
		pass
	pass


class Volume_Slider_Frame(Frame):
	def __init__(self, master=None, cnf={}, **kw):
		super().__init__(master=master, cnf=cnf, **kw)

		# Label
		self.lbl_Volume = Label(self, text='Volume', font=LBL_FONT)
		self.lbl_Volume.grid(column=0, row=0)

		# Slider (Scale)
		self.scl_Volume = Scale(self, length=200, orient=HORIZONTAL, resolution=1, font=SCL_FONT)
		self.scl_Volume.set(100)
		self.scl_Volume.grid(column=1, row=0)
		pass
	pass


class Station_Slider_Frame(Frame):
	def __init__(self, master=None, cnf={}, **kw):
		super().__init__(master=master, cnf=cnf, **kw)

		# Label
		self.lbl_Station = Label(self, text='Station', font=LBL_FONT)
		self.lbl_Station.grid(column=0, row=0)

		# Station Selection Bar
		self.scl_Station = Scale(self, length=200, command=self.Radio_Slider, orient=HORIZONTAL, resolution=0.1, from_=88.1, to=108.1, font=SCL_FONT)
		self.scl_Station.set(88.1)
		self.scl_Station.grid(column=1, row=0)
		pass

	# Ensure all selected values are odd
	def Radio_Slider(self, val):
		if (float(val)*5 % 1) * 5 == 0:
			self.scl_Station.set(float(val) + .1)
			pass
		pass
	pass


class Refresh_Frame(Frame):
	def __init__(self, master=None, cnf={}, **kw):
		super().__init__(master=master, cnf=cnf, **kw)

		# Refresh Stations Button
		self.btn_Refresh = Button(self, text='Refresh', command=lambda: station_collect_event.set(), font=BTN_FONT)
		self.btn_Refresh.grid(column=0, row=0)

		# Progress Bar
		self.prog_bar = ttk.Progressbar(self, mode='determinate', maximum=len(Radio_Listener.Radio_Data), length=200)
		self.prog_bar.grid(column=1, row=0)
		pass
	pass


class Body_Frame(Frame):
	def __init__(self, master=None, cnf={}, **kw):
		super().__init__(master=master, cnf=cnf, **kw)
		
		# Buttons
		self.Buttons = Button_Frame()
		self.Buttons.grid(row=1, column=1, rowspan=3)

		# Station Slider
		self.Stations = Station_Slider_Frame()
		self.Stations.grid(row=1, column=0)

		# Volume Slider
		self.Volume = Volume_Slider_Frame()
		self.Volume.grid(row=2, column=0)

		# Refresh Button + Progress Bar
		self.Refresh = Refresh_Frame()
		self.Refresh.grid(row=3, column=0)

		pass
	pass


class Main_Window(Tk):
	def __init__(self):
		super().__init__()

		# Window Settings
		self.title('Radio Classifier')
		self.geometry('500x210')

		# Frames
		self.frm_Body = Body_Frame(self)
		self.frm_Body.grid(column=0, row=1)

		# Labels
		self.lbl_Title = Label(self, text='Radio Classifier', font=TTL_FONT)
		self.lbl_Title.grid(column=0, row=0)
		pass
	pass


# ==============================================
# Window Instance
# ==============================================

global window; window = Main_Window()

# ==============================================
# Daemon Definitions
# ==============================================

def Station_Read_Handler():
	global window
	print('Beginning Read Handler Daemon')
	while True:
		if station_read_event.is_set():
			station_read_event.clear()
			window.prog_bar.step(1)
			pass
		pass
	pass

# ==============================================
# Daemon Threads
# ==============================================

# Steps Progressbar based on read progress
read_Handler = threading.Thread(target=Station_Read_Handler, daemon=True)
read_Handler.start()

# Setup SDR for Listener
read_Station = None
sdr = None

try:
	sdr = RtlSdr()
	sdr.set_bandwidth(.2e6) # Hz
	sdr.sample_rate = 2.4e6
	sdr.freq_correction = 60   # PPM
	sdr.gain = 0.0

	# Handles intake of new data
	read_Station = threading.Thread(target=Radio_Listener.__main__, args=(sdr, station_read_event, station_collect_event,), daemon=True)
	read_Station.start()
except:
	sdr = None
	print('Error: SDR Unable to be initialized on startup.')
	pass

# ==============================================
# Window Thread
# ==============================================

window.mainloop()

# ==============================================
# Cleanup
# ==============================================

if sdr != None:
	sdr.close()
	pass
