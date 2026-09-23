import json
import re
import subprocess
import serial.tools.list_ports as lp


def get_bluetooth_names():
    ps = subprocess.run(
        [
            'powershell',
            '-NoProfile',
            '-Command',
            'Get-PnpDevice -Class Bluetooth -ErrorAction SilentlyContinue | Select-Object -Property FriendlyName,InstanceId | ConvertTo-Json -Depth 10',
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    entries = json.loads(ps.stdout)
    names = {}

    for item in entries:
        instance_id = item.get('InstanceId', '')
        friendly_name = item.get('FriendlyName', '').strip()
        if not friendly_name:
            continue

        for match in re.finditer(r'(?:DEV|BLUETOOTHDEVICE)_([0-9A-Fa-f]+)', instance_id):
            key = match.group(1).upper()
            names[key] = friendly_name

        for match in re.finditer(r'&([0-9A-Fa-f]{12})', instance_id):
            key = match.group(1).upper()
            names[key] = friendly_name

    return names


def extract_device_key(hwid: str):
    matches = re.findall(r'&([0-9A-Fa-f]{12})', hwid)
    if matches:
        return matches[0].upper()
    return None


def main():
    bluetooth_names = get_bluetooth_names()

    print('COM ports with Windows Bluetooth-friendly names:')
    print('')
    for port in lp.comports():
        key = extract_device_key(port.hwid)
        friendly_name = bluetooth_names.get(key, port.description)
        print(f'{port.device}: {friendly_name}')


if __name__ == '__main__':
    main()
