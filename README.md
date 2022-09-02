# sky-estimation-WP3.7
Repository for sky-subtraction routines to preserve LSB features, including tests with mock data

makeDwarfs : script for making model dwarf ETGs for injection into model clusters.  See DEMO.ipynb in that directory for details.

measureMetrics : software for doing photometry on images and outputting metrics to track improvements to the LSST sky subtraction algorithm

syntheticImages : software for creating mock images, with model galaxies, stars, and noise.  Used for testing sky-subtraction algorithm against ground-truth.

RealDataTests.pdf : compiled Jupyter Notebook containing a description of our proposed novel sky-subtraction method, with examples, as well as qualitative and quantitative tests of this method done on real imaging data from two telescopes.


### License
***
Copyright 2022 Aaron Watkins, University of Hertfordshire

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
