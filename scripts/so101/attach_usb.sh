#!/usr/bin/env bash
# Usage: attach_usb.sh    Run from Git Bash on Windows or from inside WSL. Re-attaches the arms and cameras to WSL.
# A camera that drops out mid-recording stays "Shared" on Windows; attaching it again is all it needs.
# First-time binding of a new device needs an admin PowerShell: usbipd bind --busid <id>
DEVICES="${SO101_USB_IDS:-1a86:55d3 13d3:784b 05a3:9230}"

usbipd.exe list | tr -d '\r' | while read -r busid vidpid rest; do
  [[ " $DEVICES " == *" $vidpid "* ]] || continue
  case "$rest" in
    *Attached) echo "$busid $vidpid already attached" ;;
    *Shared) echo "$busid $vidpid attaching"; usbipd.exe attach --wsl --busid "$busid" ;;
    *) echo "$busid $vidpid NOT SHARED - needs: usbipd bind --busid $busid (admin PowerShell)" ;;
  esac
done
