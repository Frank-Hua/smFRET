import xml.etree.ElementTree as et

def fret_settings(path):
    settings = et.parse(path).getroot()
    assert settings.tag == 'settings'

    parsed_settings = {}
    for setting in settings:
        if setting.attrib.get("type") == "int":
            parsed_settings[setting.tag] = int(setting.text)
        elif setting.attrib.get("type") == "float":
            parsed_settings[setting.tag] = float(setting.text)
        else:
            parsed_settings[setting.tag] = setting.text

    return parsed_settings
