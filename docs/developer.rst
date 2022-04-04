Developer instructions
======================

Guidance for Firelink developers

Pre-Config and Black
--------------------

For developers, Bleeding edge dependency testing uses `pre-commit <https://pre-commit.com/>`_ to run the
`black <https://github.com/ambv/black>`_ code formatter before each commit.  In a nutshell what happens is:


* You edit some code locally, then ``git add`` and ``git commit``
* As the commit is registered by ``git``\ , the ``pre-commit`` package runs ``black`` on any changed files
  in the commit.  (You can also run ``pre-commit run --all-files`` to run on all files) and will show
  as having *failed* if it has made any edits.  If ``black`` has made edits to the edited files, you
  need to re-add and re-commit those specific files.
* Then you can ``git push`` to your hearts' content!
* The ``black`` code formatter forces our hand on code formatting and style - that means that all of
  the code follows the same style, and we can focus on the meat of the issue.

The first time you clone Firelink into a new environment, you need to run ``pre-commit install`` after
pip/conda installation.  You do *not* need pre-commit to just run Firelink.  If you have a pre-existing
environment, go ahead and ``pip install pre-commit`` and you should be fine. Generally,
only PRs with black styling will be accepted by the Firelink team.

Contribution guidelines
-----------------------

Keep an eye on the `issues <https://github.com/couyang24/Firelink/issues>`_.
We are always happy for help, including such things as:

- Bug reports
- Feature requests
- Commenting on issues ("me too!" and "+1" can be helpful)
- Positive feedback (It's always lovely to hear!)
- Negative feedback (but be nice)
- Pull Requests to fix a bug
- Pull Requests to implement a feature (though we wouldn't mind discussing first)
