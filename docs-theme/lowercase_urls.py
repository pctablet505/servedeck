"""Serve every page at a lower-case address: `OPERATIONS.md` becomes `/operations/`."""


def on_files(files, config):
    for file in files.documentation_pages():
        file.dest_uri = file.dest_uri.lower()
    return files
