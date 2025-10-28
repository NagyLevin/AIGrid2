import subprocess
import time
import sys
import os
import shutil
from typing import List, Tuple, Dict

PY = sys.executable  # ugyanaz a Python, amivel ezt a scriptet fut
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# --- LOG tisztítás kapcsoló ----------------------------------------------------
DELETE_OLD_LOGS = True      # Állítsd True-ra, ha induláskor törölni kell a logs mappát
LOG_DIR = os.path.join(BASE_DIR, "logs")
# -------------------------------------------------------------------------------

# Kimeneti könyvtár
os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)

# --- HERE CAN I ADD THE BOTS ----------------------------------------------------
# Format: ("BOT_NAME", "botlogic_file.py")
BOTS: List[Tuple[str, str]] = [
    
    ("BOT1", "lieutenant_crown_him_with_many_crowns_thy_full_gallant_legions_he_found_it_in_him_to_forgive.py"),
    #("BOT2", "naive_astar_no_speed.py"),
    #("BOT3", "astarrework.py"),
    #("BOT4", "levin.py"),
    # Példa: ("SAJAT_BOT", "my_cool_bot.py"),
]
# -------------------------------------------------------------------------------

# Judge parancs # figyelj arra hogy át kell állítani a playerek számát
judge_cmd = [
    PY, "judge/run.py", "judge/sample_config.json", "1",
    "--replay_file", "output/replay.json",
    "--output_file", "output/output.json",
    "--connection_timeout", "60",
]

# Bot parancs összeállító
def make_bot_cmd(logic_filename: str) -> List[str]:
    # A client_bridge alapértelmezetten a "logs" mappába fog írni (relatív a cwd-hez),
    # amit ez a script induláskor töröl/újralétrehoz.
    return [PY, "bot/client_bridge.py", os.path.join("bot", logic_filename)]

def spawn(name: str, cmd: List[str]) -> subprocess.Popen:
    print(f">> START {name}: {' '.join(cmd)}", flush=True)
    p = subprocess.Popen(cmd, cwd=BASE_DIR)
    time.sleep(0.5)  # ha azonnal kilép, észrevegyük
    rc = p.poll()
    if rc is not None:
        print(f"!! {name} AZONNAL LEÁLLT, exit code: {rc}", flush=True)
    return p

def check_required_files() -> None:
    required = [
        "judge/run.py",
        "judge/sample_config.json",
        "bot/client_bridge.py",
    ]
    # A megadott botlogikák is kellenek
    for _, logic in BOTS:
        required.append(os.path.join("bot", logic))

    missing = [p for p in required if not os.path.exists(os.path.join(BASE_DIR, p))]
    if missing:
        print("!! HIÁNYZÓ FÁJLOK:", *[f" - {m}" for m in missing], sep="\n")
        sys.exit(1)

def cleanup_logs_if_needed() -> None:
    if DELETE_OLD_LOGS:
        try:
            shutil.rmtree(LOG_DIR, ignore_errors=True)
            print(f">> LOGS törölve: {LOG_DIR}", flush=True)
        except Exception as e:
            print(f"!! LOGS törlés hiba: {e}", flush=True)
    # újralétrehozás (ha töröltük, vagy ha nem létezett)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        # nem kötelező, de jó látni futás közben:
        print(f">> LOGS mappa készen: {LOG_DIR}", flush=True)
    except Exception as e:
        print(f"!! LOGS mappa létrehozás hiba: {e}", flush=True)

def main() -> None:
    check_required_files()

    # LOG-ok takarítása és mappakészítés (minden indításkor ezen script kezeli)
    cleanup_logs_if_needed()

    # Judge indítása
    judge = spawn("JUDGE", judge_cmd)
    time.sleep(1.5)  # adjunk időt a szervernek felállni

    # Botok indítása a BOTS listából
    bot_procs: Dict[str, subprocess.Popen] = {}
    for bot_name, logic_file in BOTS:
        cmd = make_bot_cmd(logic_file)
        bot_procs[bot_name] = spawn(bot_name, cmd)

    try:
        rc_judge = judge.wait()
        print(f">> JUDGE LEZÁRULT, exit code: {rc_judge}", flush=True)

        # Várunk minden botra
        for name, proc in bot_procs.items():
            try:
                proc.wait()
                print(f">> {name} LEZÁRULT", flush=True)
            except Exception as e:
                print(f"!! {name} várakozás hiba: {e}", flush=True)

    except KeyboardInterrupt:
        print("\n>> CTRL+C – folyamatok leállítása...", flush=True)
    finally:
        # Leállítás fordított sorrendben: előbb a botok, majd a judge
        for name, proc in list(bot_procs.items()) + [("JUDGE", judge)]:
            try:
                if proc and proc.poll() is None:
                    proc.terminate()
                    time.sleep(0.5)
                    if proc.poll() is None:
                        proc.kill()
                print(f">> {name} LEÁLLÍTVA", flush=True)
            except Exception as e:
                print(f"!! {name} leállítás hiba: {e}", flush=True)

if __name__ == "__main__":
    main()
