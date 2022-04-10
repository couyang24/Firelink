from firelink._version import __version__

from .logger import get_logger

print(
    """
I am a Fire Keeper. I tend to the flame, and tend to thee. May the flame guide thee!

                                     g,
                                    @@@
                                   `@@$
                                    `%|
                                    @@m_
                                    @@@@@@w_
                                   )@@@@@@@@@_
                         @@        @@@@@@@@@@@k
                        @@@       @@@@@@@@@@@@@k
                        @@r     g@@@@@@M@@@@@@@@L
                        *$     @@@@@@@@@@@@@@@@@|
                             ,@@@@@@@@@@@@@@@@@@|
                         (   @@@@@@@@@@@@@@@@@@@L
                        j@  |@@@@@@@@@@@@@@@@@@@        |
                        @@$ )@@@@@@@*`  |@@@@@@@__g@$  )@k
                       |@@@@@@@@@@'     `@@@@@W@@@@@@L *@$
                       @@@@@@@@@@        '@@@@@@@M@@@$  `
                       @@@@@@@@@|          |@@@@@@@@@@
                     g_@@@@@@@@@|           |@m@@@@M@@aL
                     M@@@@@@@@`|@            | |@@@@@@@L
                     $@@@@@@@|                  @@@@@@@L
                     )@@@@@@@                   @@m@@@@
                      @@@@@@@|                  @@@@@@k
                      `@@@@@@@                 |@@@@@M
                        M@@@@@@|              g@@@@@*
                         `%@@@@@@g|       _g@@@@@M*
                             `****************"`
"""
)

LOG = get_logger(__name__)
LOG.info(f"Installing Firelink Version: {__version__}")
