#!/bin/sh

sudo launchctl load -w /Library/LaunchDaemons/org.freedesktop.dbus-system.plist

launchctl load -w /Library/LaunchAgents/org.freedesktop.dbus-session.plist

meld
