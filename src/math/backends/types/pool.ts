/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Conv2DInfo} from '../../conv_util';
import {Array4D} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputConfig, KernelNode, TapeNodeInputArrays, TapeNodeInputGradientArrays} from '../tape_config';

// Pool
export interface PoolNode extends KernelNode {
  inputAndArgs: PoolInputConfig;
  output: Array4D;
  gradient: (dy: Array4D, y: Array4D) => PoolGradientInputArrays;
}

export interface PoolInputConfig extends KernelInputConfig {
  inputs: PoolInputArrays;
  args: {convInfo: Conv2DInfo;};
}

export interface PoolInputArrays extends TapeNodeInputArrays {
  x: Array4D;
}

export interface PoolGradientInputArrays extends TapeNodeInputGradientArrays {
  x: () => Array4D;
}

// PoolBackprop
export interface PoolBackpropNode extends KernelNode {
  inputAndArgs: PoolInputConfig;
  output: Array4D;
  gradient: (dy: Array4D, y: Array4D) => PoolBackpropGradientInputArrays;
}

export interface PoolBackpropInputConfig extends KernelInputConfig {
  inputs: PoolBackpropInputArrays;
  args: {convInfo: Conv2DInfo;};
}

export interface PoolBackpropInputArrays extends TapeNodeInputArrays {
  dy: Array4D;
  x: Array4D;
}

export interface PoolBackpropGradientInputArrays extends
    TapeNodeInputGradientArrays {
  dy: () => Array4D;
  x: () => Array4D;
}
