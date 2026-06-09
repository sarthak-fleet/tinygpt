#!/usr/bin/env python3
"""build-bfcl-pace-12.py — generate Pace-12-action BFCL subset.

Writes 96 prompts (8 per action × 12 actions) in BFCL v3 format:
  - prompts file:        BFCL_v3_pace12.json  (one JSONL row / line)
  - ground-truth file:   possible_answer/BFCL_v3_pace12.json

Compatible with scripts/eval_bfcl.py existing runner.

The BFCL "possible value list" idea:
- Each arg gets a list of acceptable values. The model passes if its emitted
  arg value is contained in that list.
- For free-text fields (subject lines, body text, search queries), we set a
  semi-open list — multiple plausible synonyms. For exact things (app names,
  scroll direction enum), we set strict lists.
"""
import json
from pathlib import Path

BFCL_DIR = Path.home() / ".cache/tinygpt/datasets/bfcl"
OUT_PROMPTS = BFCL_DIR / "BFCL_v3_pace12.json"
OUT_GROUND = BFCL_DIR / "possible_answer" / "BFCL_v3_pace12.json"


# ----- function definitions ------------------------------------------------
# BFCL expects function[].parameters in JSONSchema-ish format. We summarize.
PACE_FUNCTIONS = [
    {
        "name": "AX.press",
        "description": "Click / press a UI element by its visible label.",
        "parameters": {"type": "dict",
            "properties": {"target": {"type": "string", "description": "Element label or AX identifier"}},
            "required": ["target"]}
    },
    {
        "name": "AX.setValue",
        "description": "Set the value of a text input element.",
        "parameters": {"type": "dict",
            "properties": {
                "target": {"type": "string", "description": "Label of the text field; 'focused' for whatever has focus"},
                "value":  {"type": "string", "description": "Text content to write"},
            },
            "required": ["target", "value"]}
    },
    {
        "name": "AX.scroll",
        "description": "Scroll the focused scrollable element.",
        "parameters": {"type": "dict",
            "properties": {
                "direction": {"type": "string", "enum": ["up", "down", "left", "right", "top", "bottom"]},
                "amount":    {"type": "integer"},
            },
            "required": ["direction"]}
    },
    {
        "name": "App.launch",
        "description": "Launch a macOS application.",
        "parameters": {"type": "dict",
            "properties": {"name": {"type": "string", "description": "Display name or bundle id"}},
            "required": ["name"]}
    },
    {
        "name": "App.activate",
        "description": "Bring an already-running app to the front.",
        "parameters": {"type": "dict",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]}
    },
    {
        "name": "Mail.draft",
        "description": "Open Mail compose with populated fields.",
        "parameters": {"type": "dict",
            "properties": {
                "to":      {"type": "array", "items": {"type": "string"}, "description": "Recipient email addresses or names"},
                "cc":      {"type": "array", "items": {"type": "string"}},
                "subject": {"type": "string"},
                "body":    {"type": "string"},
            },
            "required": ["to"]}
    },
    {
        "name": "Cal.event",
        "description": "Create a calendar event.",
        "parameters": {"type": "dict",
            "properties": {
                "title": {"type": "string"},
                "start": {"type": "string", "description": "ISO-8601 datetime"},
                "end":   {"type": "string"},
                "location": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["title", "start"]}
    },
    {
        "name": "Reminders.add",
        "description": "Add a Reminder.",
        "parameters": {"type": "dict",
            "properties": {
                "title": {"type": "string"},
                "due":   {"type": "string"},
                "list":  {"type": "string"},
                "priority": {"type": "string", "enum": ["none", "low", "medium", "high"]},
            },
            "required": ["title"]}
    },
    {
        "name": "Notes.create",
        "description": "Create an Apple Note.",
        "parameters": {"type": "dict",
            "properties": {
                "title":  {"type": "string"},
                "body":   {"type": "string"},
                "folder": {"type": "string"},
            },
            "required": ["body"]}
    },
    {
        "name": "Shortcut.run",
        "description": "Run a user-installed Apple Shortcut by name.",
        "parameters": {"type": "dict",
            "properties": {
                "name":  {"type": "string"},
                "input": {"type": "string"},
            },
            "required": ["name"]}
    },
    {
        "name": "Window.snap",
        "description": "Snap the focused window to a screen position.",
        "parameters": {"type": "dict",
            "properties": {
                "position": {"type": "string", "enum": ["left", "right", "top", "bottom", "fullscreen", "center", "topleft", "topright", "bottomleft", "bottomright"]},
            },
            "required": ["position"]}
    },
    {
        "name": "Clipboard.read",
        "description": "Read the current pasteboard contents.",
        "parameters": {"type": "dict", "properties": {}, "required": []}
    },
]
assert len(PACE_FUNCTIONS) == 12

FN_BY_NAME = {f["name"]: f for f in PACE_FUNCTIONS}


# ----- prompts + ground truth (8 per action × 12) --------------------------
# Each row: (action_name, prompt, args_dict_with_possible_values)
# args_dict values are lists of acceptable values (BFCL convention).
ROWS = []


# AX.press — 8
for prompt, target in [
    ("click the save button",                              ["Save", "save"]),
    ("press submit",                                       ["Submit", "submit"]),
    ("tap OK",                                             ["OK", "Ok", "ok"]),
    ("hit the cancel button",                              ["Cancel", "cancel"]),
    ("click the login button",                             ["Login", "Log In", "Sign In", "login"]),
    ("press the back arrow",                               ["Back", "back", "Back arrow"]),
    ("click delete",                                       ["Delete", "delete"]),
    ("press the search button",                            ["Search", "search"]),
]:
    ROWS.append(("AX.press", prompt, {"target": target}))


# AX.setValue — 8
for prompt, target, value in [
    ("type hello world in the message box",                ["message box", "message", "Message"],     ["hello world", "Hello world", "Hello World"]),
    ("write yes in the comment field",                     ["comment field", "comment", "Comment"],   ["yes", "Yes"]),
    ("enter john at example dot com in the email field",   ["email field", "email", "Email"],         ["john@example.com", "John@example.com"]),
    ("put 100 in the quantity input",                      ["quantity input", "quantity", "Quantity"], ["100"]),
    ("set the subject to weekly update",                   ["subject", "Subject"],                     ["weekly update", "Weekly update", "Weekly Update"]),
    ("write friday in the date field",                     ["date field", "date", "Date"],            ["friday", "Friday"]),
    ("type my password in the password field",             ["password field", "password", "Password"], ["my password", "My password"]),
    ("enter the title as final report",                    ["title", "Title"],                         ["final report", "Final report", "Final Report"]),
]:
    ROWS.append(("AX.setValue", prompt, {"target": target, "value": value}))


# AX.scroll — 8
for prompt, direction in [
    ("scroll down",                  ["down"]),
    ("scroll up",                    ["up"]),
    ("scroll all the way to top",    ["top"]),
    ("scroll to the bottom",         ["bottom"]),
    ("scroll left",                  ["left"]),
    ("scroll right a bit",           ["right"]),
    ("go back to the top of this",   ["top"]),
    ("scroll down a few times",      ["down"]),
]:
    ROWS.append(("AX.scroll", prompt, {"direction": direction}))


# App.launch — 8
for prompt, name in [
    ("open mail",                            ["Mail", "mail", "com.apple.mail"]),
    ("launch safari",                        ["Safari", "safari", "com.apple.Safari"]),
    ("start the music app",                  ["Music", "music", "com.apple.Music"]),
    ("open calendar",                        ["Calendar", "calendar", "com.apple.iCal"]),
    ("launch xcode",                         ["Xcode", "xcode", "com.apple.dt.Xcode"]),
    ("open the notes app",                   ["Notes", "notes", "com.apple.Notes"]),
    ("start slack",                          ["Slack", "slack", "com.tinyspeck.slackmacgap"]),
    ("open finder",                          ["Finder", "finder", "com.apple.finder"]),
]:
    ROWS.append(("App.launch", prompt, {"name": name}))


# App.activate — 8
for prompt, name in [
    ("switch to mail",                       ["Mail", "mail"]),
    ("bring safari to the front",            ["Safari", "safari"]),
    ("focus on slack",                       ["Slack", "slack"]),
    ("go back to xcode",                     ["Xcode", "xcode"]),
    ("activate the notes app",               ["Notes", "notes"]),
    ("switch to my browser",                 ["Safari", "Chrome", "Arc", "browser"]),
    ("focus terminal",                       ["Terminal", "terminal", "iTerm", "iTerm2"]),
    ("bring messages forward",               ["Messages", "messages"]),
]:
    ROWS.append(("App.activate", prompt, {"name": name}))


# Mail.draft — 8
for prompt, to, subject_list, body_list in [
    ("draft an email to john saying hi",                            ["john", "John", "john@example.com"], ["", "hi", "hello", "greeting"], ["hi", "Hi", "Hello", "hello"]),
    ("compose a message to sarah about lunch",                      ["sarah", "Sarah", "sarah@example.com"], ["lunch", "Lunch"], ["", "lunch", "about lunch"]),
    ("send an email to the team about the friday demo",             ["team", "team@example.com", "the team"], ["friday demo", "Friday demo", "Friday Demo", "demo"], ["", "demo", "friday demo"]),
    ("email mom that i landed safely",                              ["mom", "Mom"], ["", "landed safely", "I landed safely"], ["i landed safely", "I landed safely", "landed safely"]),
    ("write an email to bob with subject quick question",           ["bob", "Bob"], ["quick question", "Quick question", "Quick Question"], [""]),
    ("draft to alice and carol about the merger",                   ["alice", "Alice", "carol", "Carol"], ["merger", "Merger", "the merger"], [""]),
    ("compose an email to support saying my screen is frozen",      ["support", "support@example.com"], ["", "screen frozen", "Screen frozen", "frozen screen"], ["my screen is frozen", "screen frozen", "screen is frozen"]),
    ("email tom about the dinner reservation",                      ["tom", "Tom"], ["dinner reservation", "Dinner reservation", "Dinner Reservation", "dinner"], [""]),
]:
    args = {"to": [to]}
    args["subject"] = subject_list
    args["body"] = body_list
    ROWS.append(("Mail.draft", prompt, args))


# Cal.event — 8
for prompt, title_list, start_list in [
    ("create a meeting tomorrow at 3pm",                  ["meeting", "Meeting"],                  ["tomorrow 15:00", "tomorrow 3pm", "T+1 15:00"]),
    ("schedule a one-on-one with bob friday at 10am",     ["one on one with bob", "1:1 with Bob", "one-on-one with Bob"], ["friday 10:00", "Friday 10am", "Friday 10:00"]),
    ("add a calendar event for the dentist next monday 9am", ["dentist", "Dentist", "dentist appointment"], ["next monday 9:00", "Monday 9am", "next monday 09:00"]),
    ("book a 30 min sync at 2pm today",                   ["sync", "30 min sync", "30-minute sync"], ["today 14:00", "today 2pm", "2pm"]),
    ("create an event called lunch with marcus at noon",  ["lunch with marcus", "Lunch with Marcus"], ["today 12:00", "today noon", "noon"]),
    ("schedule a team standup every weekday at 9",        ["team standup", "Team standup", "standup"], ["09:00", "today 9am", "tomorrow 09:00"]),
    ("add a quarterly review on march 31 at 4pm",         ["quarterly review", "Q1 review", "Quarterly review"], ["march 31 16:00", "2026-03-31 16:00", "March 31 4pm"]),
    ("create an event for the doctor visit tuesday morning", ["doctor visit", "Doctor visit", "doctor"], ["tuesday 09:00", "tuesday morning", "Tuesday morning"]),
]:
    ROWS.append(("Cal.event", prompt, {"title": title_list, "start": start_list}))


# Reminders.add — 8
for prompt, title_list, due_list in [
    ("remind me to call mom tonight",                  ["call mom", "Call mom", "Call Mom"],        ["tonight", "Tonight", "today evening", "today 20:00"]),
    ("add a reminder to pay rent on the 1st",          ["pay rent", "Pay rent"],                    ["1st", "the 1st", "next 1st", "next month 1st"]),
    ("remind me to take out trash tomorrow morning",   ["take out trash", "Take out trash", "trash"], ["tomorrow morning", "tomorrow 08:00", "Tomorrow morning"]),
    ("set a reminder to email john on friday",         ["email john", "Email John", "email john on friday"], ["friday", "Friday"]),
    ("remind me to buy milk",                          ["buy milk", "Buy milk"],                     [""]),
    ("add a high priority reminder to file taxes by april 15", ["file taxes", "File taxes"],          ["april 15", "April 15", "2026-04-15"]),
    ("remind me to check the oven in 20 minutes",      ["check the oven", "check oven", "Check oven"], ["in 20 minutes", "20 min", "+20 min"]),
    ("remind me to follow up with the client next monday", ["follow up with the client", "follow up with client", "Follow up with client"], ["next monday", "Monday", "next Monday"]),
]:
    ROWS.append(("Reminders.add", prompt, {"title": title_list, "due": due_list}))


# Notes.create — 8
for prompt, body_list in [
    ("take a note that says project deadline is friday",       ["project deadline is friday", "Project deadline is friday", "Project deadline is Friday"]),
    ("create a note saying remember to call the bank",          ["remember to call the bank", "Remember to call the bank", "call the bank"]),
    ("note this: book club is at sarah's on saturday",          ["book club is at sarah's on saturday", "Book club is at Sarah's on Saturday"]),
    ("write a note that the wifi password is starfish123",      ["the wifi password is starfish123", "wifi password is starfish123", "WiFi password: starfish123"]),
    ("save a note saying ideas for next quarter",               ["ideas for next quarter", "Ideas for next quarter", "Ideas for Q1"]),
    ("create a note titled grocery list with milk eggs bread", ["milk eggs bread", "milk, eggs, bread", "Milk eggs bread"]),
    ("note that mom's birthday is november twelfth",            ["mom's birthday is november twelfth", "Mom's birthday is November 12", "Mom's birthday: Nov 12"]),
    ("take a note: passport expires in june",                   ["passport expires in june", "Passport expires in June", "passport: expires June"]),
]:
    ROWS.append(("Notes.create", prompt, {"body": body_list}))


# Shortcut.run — 8
for prompt, name_list in [
    ("run my morning routine shortcut",      ["morning routine", "Morning Routine", "Morning routine"]),
    ("trigger the focus mode shortcut",      ["focus mode", "Focus Mode", "Focus mode"]),
    ("run the home arrival shortcut",        ["home arrival", "Home Arrival", "Home arrival"]),
    ("execute my workout starter",           ["workout starter", "Workout Starter", "Workout starter"]),
    ("run the convert image to webp shortcut", ["convert image to webp", "Convert Image to WebP", "Convert image to WebP"]),
    ("trigger my bedtime shortcut",          ["bedtime", "Bedtime"]),
    ("run focus deep work",                  ["focus deep work", "Focus Deep Work", "Deep work focus", "Deep work"]),
    ("start the daily standup shortcut",     ["daily standup", "Daily Standup", "Daily standup"]),
]:
    ROWS.append(("Shortcut.run", prompt, {"name": name_list}))


# Window.snap — 8
for prompt, position_list in [
    ("snap this window to the left",       ["left"]),
    ("move this window to the right half", ["right"]),
    ("make this fullscreen",               ["fullscreen"]),
    ("center the window",                  ["center"]),
    ("snap to top left",                   ["topleft"]),
    ("send this to the bottom right",      ["bottomright"]),
    ("dock the window to the top half",    ["top"]),
    ("snap this to the bottom half",       ["bottom"]),
]:
    ROWS.append(("Window.snap", prompt, {"position": position_list}))


# Clipboard.read — 8
for prompt in [
    "read my clipboard",
    "what's on the clipboard",
    "tell me what i copied",
    "say my clipboard contents out loud",
    "what's in my paste buffer",
    "check the clipboard for me",
    "give me what's copied",
    "speak the clipboard contents",
]:
    ROWS.append(("Clipboard.read", prompt, {}))


# --- build BFCL records ----------------------------------------------------
def build():
    BFCL_DIR.mkdir(parents=True, exist_ok=True)
    (BFCL_DIR / "possible_answer").mkdir(parents=True, exist_ok=True)

    prompts = []
    truth = []
    counts = {}
    for i, (fn, prompt, args) in enumerate(ROWS):
        rid = f"pace12_{i:03d}_{fn.replace('.', '_').lower()}"
        prompts.append({
            "id": rid,
            "question": [[{"role": "user", "content": prompt}]],
            "function": [FN_BY_NAME[fn]],
        })
        truth.append({
            "id": rid,
            "ground_truth": [{fn: args}],
        })
        counts[fn] = counts.get(fn, 0) + 1

    OUT_PROMPTS.write_text("\n".join(json.dumps(r) for r in prompts) + "\n")
    OUT_GROUND.write_text("\n".join(json.dumps(r) for r in truth) + "\n")

    print(f"Wrote {len(prompts)} prompts to {OUT_PROMPTS}")
    print(f"Wrote {len(truth)} truth   to {OUT_GROUND}")
    print(f"Per-action counts:")
    for fn in [f["name"] for f in PACE_FUNCTIONS]:
        print(f"  {fn:20s} {counts.get(fn, 0)}")


if __name__ == "__main__":
    build()
