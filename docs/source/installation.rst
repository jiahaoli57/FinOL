Installation
============

``FinOL`` is available on `PyPI <https://pypi.org/project/finol>`__,
we recommend to install ``FinOL`` via pip:

.. code-block:: bash

   $ pip install --upgrade finol

You can also install the development version of ``FinOL``
from master branch of Git repository:

.. code-block:: bash

    $ pip install git+https://github.com/jiahaoli57/finol.git

Install ta-lib dependency
-------------------------

``FinOL`` requires the ``ta-lib`` library as one of its dependencies. ``ta-lib`` is an open-source technical indicator
library for financial applications, and unlike other dependencies, ``ta-lib`` cannot be installed automatically via ``pip install finol``.
Therefore, you will need to manually install ``ta-lib``.

Follow the steps below based on your operating system to install ``ta-lib``:

.. hint::

     For users using Anaconda, it is highly recommended to prioritize installing ``ta-lib`` using conda, as conda installation is very convenient.

.. tabs::

  .. group-tab:: pip

    .. tabs::

      .. group-tab:: Windows

          Try installing ``ta-lib`` via pip first

          .. code-block:: bash

            $ pip install TA-Lib

          If an error occurs, then install package corresponding to your Python manually from the
          `talib whl collection website <https://sourceforge.net/projects/talib-whl/files/ta_lib_0.4.28/>`__

          .. list-table::
             :header-rows: 1
             :class: ghost

             * - Python Version
               - ta-lib WHL File
             * - 3.6
               - TA_Lib-0.4.28-cp36-cp36m-win_amd64.whl
             * - 3.7
               - TA_Lib-0.4.28-cp37-cp37m-win_amd64.whl
             * - 3.8
               - TA_Lib-0.4.28-cp38-cp38-win_amd64.whl
             * - 3.9
               - TA_Lib-0.4.28-cp39-cp39-win_amd64.whl
             * - 3.10
               - TA_Lib-0.4.28-cp310-cp310-win_amd64.whl
             * - 3.11
               - TA_Lib-0.4.28-cp311-cp311-win_amd64.whl
             * - 3.12
               - TA_Lib-0.4.28-cp312-cp312-win_amd64.whl

          For example, with Python 3.10 and the whl file downloaded to the local path C:\\Users\\John\\Downloads.
          After downloading the whl file, all you need to do is run the following command:

          .. code-block:: bash

            $ pip install C:\Users\John\Downloads\TA_Lib-0.4.28-cp310-cp310-win_amd64.whl

          To verify if the installation is successful, run:

          .. code-block:: bash

            $ python -c "import talib; print(talib.__version__)"

          If the installation is successful, it should output the current version of ``ta-lib`` without any errors.

      .. group-tab:: Linux

          Try installing ``ta-lib`` via pip first

          .. code-block:: bash

            $ pip install TA-Lib

          If an error occurs, manually compile and install ``ta-lib`` from source:

          .. code-block:: bash

            $ wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
            $ tar xvzf ta-lib-0.4.0-src.tar.gz
            $ cd ta-lib/
            $ ./configure --prefix=/usr

          If you encounter an error like `configure: error: no acceptable C compiler found in $PATH`, install a C
          compiler using the appropriate command for your Linux distribution:

          - **Ubuntu / Debian**:
             .. code-block:: bash

                $ sudo apt update
                $ sudo apt install build-essential

          - **Fedora**:
              .. code-block:: bash

                $ sudo dnf groupinstall 'Development Tools'

          - **CentOS / RHEL**:
              .. code-block:: bash

                $ sudo yum groupinstall 'Development Tools'

          - **Arch Linux**:
              .. code-block:: bash

                $ sudo pacman -S base-devel

          - **openSUSE**:
              .. code-block:: bash

                $ sudo zypper install -t pattern devel_C_C++

          Then continue with:

          .. code-block:: bash

            $ make
            $ sudo make install
            $ cd ..
            $ pip install TA-Lib

          To verify if the installation is successful, run:

          .. code-block:: bash

            $ python -c "import talib; print(talib.__version__)"

          If the installation is successful, it should output the current version of ``ta-lib`` without any errors.

      .. group-tab:: Mac OSX

        .. todo::

           Will be completed later.


  .. group-tab:: conda

    .. tabs::

        .. group-tab:: All systems

          If you are using Anaconda, installation becomes very simple at this point.
          With just one line of command, you can complete the installation on 64-bit Windows, Linux and Mac OSX systems,
          including Macs with Apple M1/M2 chips using the Arm architecture:

              .. code-block:: bash

                  $ conda install -c conda-forge ta-lib -y

          To verify if the installation is successful, run:

          .. code-block:: bash

            $ python -c "import talib; print(talib.__version__)"

          If the installation is successful, it should output the current version of ``ta-lib`` without any errors.
