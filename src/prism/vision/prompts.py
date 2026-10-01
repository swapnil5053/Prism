"""Detection prompts. Kept apart from qwen.py so the eval CLI can list them
without importing torch."""

# v1 gave good boxes but called most links and buttons "text" (see eval/README.md).
# v2 defines each label and says where the box goes.
PROMPTS = {
    "v1": (
        "Detect every user interface element in this screenshot. "
        'Output a JSON array only. Each item: {"bbox_2d": [x1, y1, x2, y2], '
        '"label": one of "button", "link", "input", "checkbox", "icon", "text", "image", '
        '"text": the visible text, or "" if none}. '
        "Use one item per element. Do not group several elements into one box."
    ),
    "v2": (
        "List every user interface element in this screenshot as a JSON array. "
        'Each item: {"bbox_2d": [x1, y1, x2, y2], "label": ..., "text": visible text or ""}.\n'
        "Labels:\n"
        "- button: a clickable control drawn with its own background or border. "
        "Box the whole button shape, not just the words on it.\n"
        "- link: clickable text without a button shape, such as items in a navigation bar.\n"
        "- input: a text field or dropdown, even if it is empty. Box the whole field.\n"
        "- checkbox: a checkbox, radio button or switch.\n"
        "- icon: a small symbol that can be clicked, such as a close, menu or settings icon.\n"
        "- image: a photo, illustration, logo, or a placeholder block where an image goes.\n"
        "- text: any other text, such as headings, labels and paragraphs.\n"
        "Words that belong to a button or link are part of it; don't list them again as text. "
        "Output only the JSON array."
    ),
    # v2 folded form labels into their inputs, which broke the visible-label check.
    "v3": (
        "List every user interface element in this screenshot as a JSON array. "
        'Each item: {"bbox_2d": [x1, y1, x2, y2], "label": ..., "text": visible text or ""}.\n'
        "Labels:\n"
        "- button: a clickable control drawn with its own background or border. "
        "Box the whole button shape, not just the words on it.\n"
        "- link: clickable text without a button shape, such as items in a navigation bar.\n"
        "- input: a text field or dropdown, even if it is empty. Box only the field itself; "
        'its "text" is the placeholder inside it, if any.\n'
        "- checkbox: a checkbox, radio button or switch. Box only the small control.\n"
        "- icon: a small symbol that can be clicked, such as a close, menu or settings icon.\n"
        "- image: a photo, illustration, logo, or a placeholder block where an image goes.\n"
        "- text: any other text, such as headings, paragraphs, and the labels next to "
        "inputs and checkboxes.\n"
        "Words drawn on a button or link are part of it; don't list them again as text. "
        "A label beside or above an input or checkbox is a separate text item. "
        "Output only the JSON array."
    ),
    # For the OCR pipeline (vision/hybrid.py): OCR supplies the text, so the model
    # only lists controls and images, and doesn't spend tokens writing out words.
    "v4": (
        "List the controls and images in this screenshot as a JSON array. "
        'Each item: {"bbox_2d": [x1, y1, x2, y2], "label": ...}.\n'
        "Labels:\n"
        "- button: a clickable control drawn with its own background or border. "
        "Box the whole button shape, not just the words on it.\n"
        "- link: clickable text without a button shape, such as items in a navigation bar.\n"
        "- input: a text field or dropdown, even if it is empty. Box only the field itself.\n"
        "- checkbox: a checkbox, radio button or switch. Box only the small control.\n"
        "- icon: a small symbol that can be clicked, such as a close, menu or settings icon.\n"
        "- image: a photo, illustration, logo, or a filled block where an image goes.\n"
        "Don't list plain text such as headings, paragraphs or labels. "
        "Output only the JSON array."
    ),
}
# Default when the model also finds text (the eval CLI's --text model). The worker
# defaults to v4 with OCR text; see config.py.
DEFAULT_PROMPT = "v2"
