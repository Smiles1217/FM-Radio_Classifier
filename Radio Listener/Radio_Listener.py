from rtlsdr import RtlSdr
import numpy
import time
import threading
#import librosa
#import librosa.display
#import matplotlib.pyplot as plt

# How to generate a mel spectrogram from this FM_Station.Data:
"""
import librosa
import matplotlib.pyplot as plt

D = librosa.amplitude_to_db(numpy.abs(librosa.stft(y=FM_Station.Data)), ref=numpy.max)
librosa.display.specshow(D)
"""
class FM_Station:
	def __init__(self, freq = 88.1e6):
		self.Frequency = freq # Hz - Center frequency of the station
		self.Data = []        # Partially processed data values
		self.IsAvail = True   # Flag indicating if data is available for this station, default True, toggled off during initialization
		self.IsActive = True  # Flag indicating if station is considered Active, default True
		self.Means = []       # Historical data of means of each .375s sample
		self.Maxs = []        # Historical data of maxs of each .375s sample

	# Give lock
	def Get_Lock(self):
		if self.IsAvail:  # If available, give lock
			self.IsAvail = False
			return True
		return False # Otherwise, ignore

	def Sample(self, sdr):
		self.Data = []  # Clear old data
		sdr.center_freq = self.Frequency  # Target station
		self.Data = numpy.abs(sdr.read_samples(Sample_Period * sdr.sample_rate))

		# Read Historical Data
		self.Means.append(numpy.mean(self.Data))
		self.Maxs.append(numpy.max(self.Data))

	# Take Lock Back
	def Release_Lock(self):
		# TODO: Add functionality to check caller for match with Get_Lock() caller
		self.IsAvail = True

	# Return FM station
	def Get_FM(self):
		return self.Frequency / 1e6


# Define Station Globals
global Sample_Period; Sample_Period = 0.375 # Sec
global Sample_Rate; Sample_Rate = 2.4e6  # Hz
global Radio_Data; Radio_Data = []
# Generate all possible radio stations
print('Generating Station Containers')
for station in numpy.arange(88.1e6, 108.2e6, .2e6):
	Radio_Data.append(FM_Station(station))
print('Stations Containers Successfully Generated')


def Collect_Data(sdr, event):
	global Radio_Data

	# Increment through the stations
	for station in Radio_Data:
		if station.IsActive: # If station is active

			if not station.IsAvail: # If station currently unavailable
				start = time.time()
				print('Awaiting station', station.Frequency/1e6, 'FM')
				while station.Get_Lock() == False: # Wait until others are done
					pass
				end = time.time()
				print('Station available after', start - end, 'seconds')
			print('Reading', station.Get_FM(), 'FM')

			# Set Station Read Event if Provided
			if event != None:
				while event.is_set():
					pass
				event.set()

			station.Sample(sdr)
			station.Release_Lock()  # Release the Lock

# Function to find all available stations
def Determine_Available_Stations():
	global Radio_Data
	# TODO: Implement active/inactive station detection
	#       If a station is deemed inactive, release its data
	for station in Radio_Data:
		pass
		# Do something.  Probably something like...
		# if max(station.Max) - max(station.Means) < Value: disable();

	print('Add inactive stations deactivated.')


# Main Program
def __main__(sdr=None, read_event=None, collect_event=None):
	global Radio_Data

	# Initialize SDR Component
	if sdr == None:
		sdr = RtlSdr()
		sdr.set_bandwidth(.2e6) # Hz
		sdr.sample_rate = 2.4e6
		sdr.freq_correction = 60   # PPM
		sdr.gain = 0.0

	# Load initial data for all stations
	start = time.time()
	Collect_Data(sdr, read_event)
	end = time.time()
	print('All stations initialized in', end - start, 'seconds')

	# Eliminate Dead Stations
	# TODO: Re-enable call once it actually works
	# Determine_Available_Stations()

	# Clear event prior to its availability
	if collect_event != None:
		collect_event.clear()

	# Infinite Runtime Loop
	while True:
		# TODO: Add interface for messages (remove station from search, terminate, ect)
		if collect_event != None and collect_event.is_set():
			Collect_Data(sdr, read_event)
			collect_event.clear()