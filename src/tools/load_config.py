import configparser


def load_config_and_initialize_class(config_file, section, cls):
    config = configparser.ConfigParser()
    config.read(config_file)

    if section not in config:
        raise ValueError(f"Section {section} not found in the config file.")

    params = {key: _convert_type(value) for key, value in config[section].items()}

    instance = cls(**params)
    return instance


def _convert_type(value):
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value
