class SubsetMixin:
    _parser_select_actions_added = False

    @classmethod
    def add_argparse_args(cls, parser, **kwargs):
        parser = super().add_argparse_args(parser, **kwargs)
        if not cls._parser_select_actions_added:
            for parser_group in parser._action_groups:
                if parser_group.title == "DatasetLoader specific arguments":
                    break
            parser_group.add_argument(
                "--select_actions",
                type=int,
                nargs="+",
                help="If given an integer selects only the first n actions. "
                "If given a list of integers selects exactly the listed "
                "actions (zero indexed). Selected actions are re-indexed in "
                "the order of the list.")
        return parser

    def __init__(self, select_actions, **kwargs):
        """
        If action subset is selected, adjust the actions list accordingly

        Parameters
        ----------
        select_actions : list of ints, optional (default is None)
            If given a list with a single integer only the first n actions are
            loaded.
            If given a list of integers exactly the listed actions are
            loaded (zero indexed). Selected actions are re-indexed in the order
            of the list.
            If not given all 60 or 120 actions are loaded.
        """
        if select_actions is not None:
            if len(select_actions) == 1:
                self.actions = [
                    self.actions[i] for i in range(select_actions[0])
                ]
            else:
                self.actions = [self.actions[i] for i in select_actions]
        super().__init__(**kwargs)

    def select_action(self, action_id, select_actions):
        """
        Skip actions which have not been selected.

        Returns a possibly re-indexed action id, or None if the action is to be
        skipped.

        Parameters
        ----------
        action_id : int
            id of the action class of the current sample
        select_actions: list of ints
            Value of the command line argument

        Returns
        -------
        Int or None
            Potentially re-indexed (if a non-trivial subset was selected)
            action id or None if the given sample should be skipped
        """
        if select_actions is None:
            return action_id
        new_action_id = None
        if len(select_actions) == 1:
            if action_id < select_actions[0]:
                new_action_id = action_id
        else:
            for i in range(len(select_actions)):
                if action_id == select_actions[i]:
                    new_action_id = i
                    break
        return new_action_id
