import omni.kit.app
import carb.settings
import carb.tokens
import os
import subprocess


def get_extension_name() -> str:
    """
    Return the name of the Extension where the module is defined.
    Args:
        None
    Returns:
        str: The name of the Extension where the module is defined.
    """
    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_id = extension_manager.get_extension_id_by_module(__name__)
    extension_name = extension_id.split("-")[0]
    return extension_name


def get_models_directory() -> str:
    extension_name = get_extension_name()
    models_directory_name = carb.settings.get_settings().get_as_string(
        f"exts/{extension_name}/models_directory"
    )
    temp_kit_directory = carb.tokens.get_tokens_interface().resolve("${data}")
    models_directory = os.path.join(temp_kit_directory, models_directory_name)
    return models_directory


def is_node_installed():
    try:
        subprocess.run(
            ["node", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
