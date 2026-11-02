"""Base class for XLA SavedModel archive."""


class XLASavedModelArchive:
    def __init__(self):
        raise NotImplementedError(
            "XLASavedModelArchive is an abstract class. "
            "Use a subclass such as SavedModelExportArchive."
        )

    def track(self, resource):
        raise NotImplementedError()

    def add_endpoint(self, name, fn, input_signature=None, **kwargs):
        raise NotImplementedError()

    def track_and_add_endpoint(self, name, resource, input_signature, **kwargs):
        raise NotImplementedError()

    def add_variable_collection(self, name, variables):
        raise NotImplementedError()

    def write_out(self, filepath, options=None, verbose=True):
        raise NotImplementedError()
