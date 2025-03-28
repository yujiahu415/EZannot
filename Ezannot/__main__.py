import requests
from packaging import version
from EZannot import __version__,gui


def main():

	try:

		current_version=version.parse(__version__)
		pypi_json=requests.get('https://pypi.org/pypi/EZannot/json').json()
		latest_version=version.parse(pypi_json['info']['version'])

		if latest_version>current_version:
			print('A newer version '+'('+str(latest_version)+')'+' of EZannot is available.')
			print('You may upgrade it by "python3 -m pip install --upgrade EZannot".')
			print('For the details of new changes, check: "https://github.com/yujiahu415/EZannot".')

	except:
		
		pass

	gui.main_window()


if __name__=='__main__':

	main()

