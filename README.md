# Edge Impulse Data Ingestion Omniverse Extension

This Omniverse extension allows you to upload your synthetic datasets to your Edge Impulse project for computer vision tasks, validate your trained model locally, and view inferencing results directly in your Omniverse synthetic environment.

![preview.png](/exts/edgeimpulse.dataingestion/data/preview.png)

## Installation

### Prerequities

* Install [Omniverse Launcher](https://docs.omniverse.nvidia.com/launcher/latest/installing_launcher.html)
* Install Isaac Sim from the Omniverse launcher
* For Windows installation, make sure to install the [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) and select the Desktop Development with C++ and NodeJS. A restart of your computer may be required to apply the changes.
* Download this repository

### Setup

* From this repository, execute the `link_app` script.

![Link app](/exts/edgeimpulse.dataingestion/data/execute-link-app.png)

* In Isaac Sim, open the Extensions view.

![Open extensions](/exts/edgeimpulse.dataingestion/data/isaac-sim-open-extensions.png)

* Open the extensions settings.

![Extension settings](/exts/edgeimpulse.dataingestion/data/isaac-sim-extensions-settings.png)

* Add the Edge Impulse extension path. Note that the extension root folder is located in the `/exts` directory.

![Extension path](/exts/edgeimpulse.dataingestion/data/isaac-sim-extension-path.png)

* Enable the Edge Impulse extension.

![Enable Edge Impulse extension](/exts/edgeimpulse.dataingestion/data/isaac-sim-enable-edgeimpulse-ext.png)


## Extension Project Template

This project was automatically generated.

- `app` - It is a folder link to the location of your *Omniverse Kit* based app.
- `exts` - It is a folder where you can add new extensions. It was automatically added to extension search path. (Extension Manager -> Gear Icon -> Extension Search Path).

Open this folder using Visual Studio Code. It will suggest you to install few extensions that will make python experience better.

Look for "edgeimpulse.dataingestion" extension in extension manager and enable it. Try applying changes to any python files, it will hot-reload and you can observe results immediately.

Alternatively, you can launch your app from console with this folder added to search path and your extension enabled, e.g.:

```
> app\omni.code.bat --ext-folder exts --enable edgeimpulse.dataingestion
```

## App Link Setup

If `app` folder link doesn't exist or broken it can be created again. For better developer experience it is recommended to create a folder link named `app` to the *Omniverse Kit* app installed from *Omniverse Launcher*. Convenience script to use is included.

Run:

```
> link_app.bat
```

If successful you should see `app` folder link in the root of this repo.

If multiple Omniverse apps is installed script will select recommended one. Or you can explicitly pass an app:

```
> link_app.bat --app create
```

You can also just pass a path to create link to:

```
> link_app.bat --path "C:/Users/bob/AppData/Local/ov/pkg/create-2021.3.4"
```


## Sharing Your Extensions

This folder is ready to be pushed to any git repository. Once pushed direct link to a git repository can be added to *Omniverse Kit* extension search paths.

Link might look like this: `git://github.com/[user]/[your_repo].git?branch=main&dir=exts`

Notice `exts` is repo subfolder with extensions. More information can be found in "Git URL as Extension Search Paths" section of developers manual.

To add a link to your *Omniverse Kit* based app go into: Extension Manager -> Gear Icon -> Extension Search Path


## Troubleshooting

**Requests**

While working in Composer, add the following snippet:

 
```
[python.pipapi]

requirements = [
    "requests"
]

use_online_index = true
```

**OSError: [WinError 126]**

If you are experiencing the `OSError: [WinError 126] The specified module could not be found. Error loading "C:\path\to\omni.isaac.ml_archive\pip_prebundle\torch\lib\fbgemm.dll" or one of its dependencies.` error, it is a recent known issue from the IsaacSim team, and the solution is to install the C/C++ build tools from VSCode 2022, here's the steps:

1. Solution description from NVIDIA: https://docs.omniverse.nvidia.com/isaacsim/latest/known_issues.html#general
2. You can install VSCode 2022 from here: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
3. Make sure to select the desktop developent C/C++ build tools in the installation screen
4. Startup VSCode 2022, and restart the Omniverse/IsaacSim apps
5. Now retry installing the Edge Impulse Omniverse extension and it should be successful!

![Install C/C++ build tools](/exts/edgeimpulse.dataingestion/data/install-development-tools-vs-code.png)

## Contributing
The source code for this repository is provided as-is and we are not accepting outside contributions.
