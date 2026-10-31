# Shoe Odds Calculator: Chrome extension

Opens the Shoe Odds Calculator in Chrome's side panel, next to the table you're watching. You enter the cards yourself. The extension doesn't read or touch any website.

## Install (Chrome or Edge, version 116 or later)

1. Download this `extension` folder, or unzip `shoe-odds-extension.zip`.
2. Open `chrome://extensions` (in Edge: `edge://extensions`).
3. Turn on **Developer mode** (top right).
4. Click **Load unpacked** and choose the `extension` folder.
5. Pin the extension from the puzzle-piece menu, then click its icon to open the side panel.

## Permissions

- `sidePanel` only. No access to websites, tabs or browsing data, and no network requests.
- Your shoe is saved in the extension's own local storage on this computer.

## Updating

The panel is generated from `prototype/shoe-odds.html`. After changing the prototype, run:

```
python3 tools/build_extension.py
```

Then click the reload icon on the extension's card in `chrome://extensions`.
