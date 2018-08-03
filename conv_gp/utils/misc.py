from gpflow.actions import Action

class RunOpAction(Action):
    def __init__(self, op):
        self.op = op

    def run(self, context):
        context.session.run(self.op)

