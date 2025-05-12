
            @echo off
            cd /d %~dp0
            if exist venv (
                call venv\Scripts\activate.bat
                venv\Scripts\python.exe streamdiffusionTD\main_sdtd.py
            ) else (
                call .venv\Scripts\activate.bat
                .venv\Scripts\python.exe streamdiffusionTD\main_sdtd.py
            )
            pause
            