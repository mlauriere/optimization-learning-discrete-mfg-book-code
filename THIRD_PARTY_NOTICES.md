# Third-Party Software and Credits

The original wrapper, plotting and test code is by Mathieu Laurière and Xin Guo
and is distributed under the MIT License in `LICENSE`.

OpenSpiel supplies the native games, learning algorithms, distribution routines
and value-function interfaces:

- Project: https://github.com/google-deepmind/open_spiel
- Tested source version: https://github.com/google-deepmind/open_spiel/tree/v1.5
- License: https://github.com/google-deepmind/open_spiel/blob/v1.5/LICENSE
- Paper: https://arxiv.org/abs/1908.09453

OpenSpiel is licensed under the Apache License, Version 2.0. Its relevant Python
sources include the notice "Copyright 2019 DeepMind Technologies Limited".
Those copyrights and license terms remain with their owners. OpenSpiel is
installed separately; its implementation sources and compiled libraries are
not redistributed in this repository. Our MIT license does not relicense it.

The concert-room geometry and Braess-network experiment settings were adapted
from the authors' research/tutorial notebooks using OpenSpiel. The routing
network helper in that material credited DeepMind Technologies Limited; this
package calls the separately installed OpenSpiel helper. Local crowd reward/
value and routing subclasses are described in `MODEL_NOTES.md` and are not
presented as changes to the upstream library.

NumPy, SciPy, Matplotlib, pandas, seaborn, pytest and other installed dependencies
retain their respective licenses. No third-party notebook, manuscript, dataset
or game-engine binary is bundled. The included reference data are numerical
outputs generated for these examples, not resumable training checkpoints.
