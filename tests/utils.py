from keyring.backend import KeyringBackend


class InMemoryKeyring(KeyringBackend):
    """
    A simple in-memory keyring backend for testing purposes.
    """

    priority = 1  # High priority to ensure it is used during tests

    def __init__(self):
        self.store = {}

    def get_password(self, service, username):
        return self.store.get((service, username))

    def set_password(self, service, username, password):
        self.store[(service, username)] = password

    def delete_password(self, service, username):
        if (service, username) in self.store:
            del self.store[(service, username)]
        else:
            raise KeyError("No such password")
