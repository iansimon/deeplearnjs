/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// tslint:disable-next-line:max-line-length
import {Array1D, Array2D, CheckpointLoader, NDArray, NDArrayMathGPU, Scalar} from 'deeplearn';
import * as demo_util from '../util';

import {KeyboardElement} from './keyboard_element';

// tslint:disable-next-line:no-require-imports
const Piano = require('tone-piano').Piano;

let lstmKernel1: Array2D;
let lstmBias1: Array1D;
//let lstmKernel2: Array2D;
//let lstmBias2: Array1D;
//let lstmKernel3: Array2D;
//let lstmBias3: Array1D;
let c: Array2D[];
let h: Array2D[];
//let inputProjBiases: Array1D;
//let inputProjWeights: Array2D;
let fullyConnectedBiases: Array1D;
let fullyConnectedWeights: Array2D;
const forgetBias = Scalar.new(1.0);
const activeNotes = new Map<number, number>();

// How many steps to generate per generateStep call.
// Generating more steps makes it less likely that we'll lag behind in note
// generation. Generating fewer steps makes it less likely that the browser UI
// thread will be starved for cycles.
const STEPS_PER_GENERATE_CALL = 10;
// How much time to try to generate ahead. More time means fewer buffer
// underruns, but also makes the lag from UI change to output larger.
const GENERATION_BUFFER_SECONDS = .5;
// If we're this far behind, reset currentTime time to piano.now().
const MAX_GENERATION_LAG_SECONDS = 1;
// If a note is held longer than this, release it.
const MAX_NOTE_DURATION_SECONDS = 3;

const CHORD_PROGRESSION_SIZE = Scalar.new(8);
const CHORDS_PER_BAR = Scalar.new(2);
const CHORD_PROGRESSION_BARS = Scalar.new(CHORD_PROGRESSION_SIZE.get() / CHORDS_PER_BAR.get());
const CHORD_ENCODING_SIZE = 49;

const QPM = 120.0;
const QUARTERS_PER_BAR = Scalar.new(4);
const DIVISIONS_PER_QUARTER = Scalar.new(24);
const BARS_PER_MINUTE = Scalar.new(QPM / QUARTERS_PER_BAR.get());

let chordProgressionIndices = Array1D.zeros([CHORD_PROGRESSION_SIZE.get()]);

let currentPianoTimeSec = 0;
// When the piano roll starts in browser-time via performance.now().
let pianoStartTimestampMs = 0;

let currentVelocity = 100;
const BASS_VELOCITY = 100;

const MIN_MIDI_PITCH = 0;
const MAX_MIDI_PITCH = 127;
const VELOCITY_BINS = 8;
const MAX_SHIFT_STEPS = 100;

const STEPS_PER_SECOND = 100;
const STEPS_PER_MINUTE = Scalar.new(60 * STEPS_PER_SECOND);

const MIDI_EVENT_ON = 0x90;
const MIDI_EVENT_OFF = 0x80;
const MIDI_NO_OUTPUT_DEVICES_FOUND_MESSAGE = 'No midi output devices found.';
const MIDI_NO_INPUT_DEVICES_FOUND_MESSAGE = 'No midi input devices found.';

// The unique id of the currently scheduled setTimeout loop.
let currentLoopId = 0;

const EVENT_RANGES = [
  ['note_on', MIN_MIDI_PITCH, MAX_MIDI_PITCH],
  ['note_off', MIN_MIDI_PITCH, MAX_MIDI_PITCH],
  ['time_shift', 1, MAX_SHIFT_STEPS],
  ['velocity_change', 1, VELOCITY_BINS],
];

function calculateEventSize(): number {
  let eventOffset = 0;
  for (const eventRange of EVENT_RANGES) {
    const minValue = eventRange[1] as number;
    const maxValue = eventRange[2] as number;
    eventOffset += maxValue - minValue + 1;
  }
  return eventOffset;
}

const EVENT_SIZE = calculateEventSize();
const SHIFT_OFFSET_LO = Scalar.new(255);
const SHIFT_OFFSET_HI = Scalar.new(356);
const MAX_SHIFT_IDX = 355;  // shift 1s.
let lastSample = Scalar.new(MAX_SHIFT_IDX);
let currentStep = Scalar.new(0);

const container = document.querySelector('#keyboard');
const keyboardInterface = new KeyboardElement(container);

const piano = new Piano({velocities: 4}).toMaster();

const SALAMANDER_URL = 'https://storage.googleapis.com/learnjs-data/' +
    'Piano/Salamander/';
const CHECKPOINT_URL = '.';

const isDeviceSupported = demo_util.isWebGLSupported();

if (!isDeviceSupported) {
  document.querySelector('#status').innerHTML =
      'We do not yet support your device. Please try on a desktop ' +
      'computer with Chrome/Firefox, or an Android phone with WebGL support.';
} else {
  start();
}

const math = new NDArrayMathGPU();

function start() {
  piano.load(SALAMANDER_URL)
      .then(() => {
        const reader = new CheckpointLoader(CHECKPOINT_URL);
        return reader.getAllVariables();
      })
      .then((vars: {[varName: string]: NDArray}) => {
        document.querySelector('#status').classList.add('hidden');
        document.querySelector('#controls').classList.remove('hidden');
        document.querySelector('#keyboard').classList.remove('hidden');

        //inputProjBiases = vars['fully_connected/biases'] as Array1D;
        //inputProjWeights = vars['fully_connected/weights'] as Array2D;

        lstmKernel1 =
            vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'] as Array2D;
        lstmBias1 =
            vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias'] as Array1D;

        //lstmKernel2 =
        //    vars['rnn/multi_rnn_cell/cell_1/lstm_cell/kernel'] as Array2D;
        //lstmBias2 =
        //    vars['rnn/multi_rnn_cell/cell_1/lstm_cell/bias'] as Array1D;

        //lstmKernel3 =
        //    vars['rnn/multi_rnn_cell/cell_2/lstm_cell/kernel'] as Array2D;
        //lstmBias3 =
        //    vars['rnn/multi_rnn_cell/cell_2/lstm_cell/bias'] as Array1D;

        //fullyConnectedBiases = vars['fully_connected_1/biases'] as Array1D;
        //fullyConnectedWeights = vars['fully_connected_1/weights'] as Array2D;

        fullyConnectedBiases = vars['fully_connected/biases'] as Array1D;
        fullyConnectedWeights = vars['fully_connected/weights'] as Array2D;

        updateChordProgression(preset1);
      });
}

function resetRnn() {
  c = [
    Array2D.zeros([1, lstmBias1.shape[0] / 4]),
    //Array2D.zeros([1, lstmBias2.shape[0] / 4]),
    //Array2D.zeros([1, lstmBias3.shape[0] / 4]),
  ];
  h = [
    Array2D.zeros([1, lstmBias1.shape[0] / 4]),
    //Array2D.zeros([1, lstmBias2.shape[0] / 4]),
    //Array2D.zeros([1, lstmBias3.shape[0] / 4]),
  ];
  if (lastSample != null) {
    lastSample.dispose();
  }
  lastSample = Scalar.new(MAX_SHIFT_IDX);
  if (currentStep != null) {
    currentStep.dispose();
  }
  currentStep = Scalar.new(0);
  currentPianoTimeSec = piano.now();
  pianoStartTimestampMs = performance.now() - currentPianoTimeSec * 1000;
  currentLoopId++;
  generateStep(currentLoopId);
}

window.addEventListener('resize', resize);

function resize() {
  keyboardInterface.resize();
}

resize();

const gainSliderElement = document.getElementById('gain') as HTMLInputElement;
const gainDisplayElement =
    document.getElementById('gain-display') as HTMLSpanElement;
let globalGain = +gainSliderElement.value;
gainDisplayElement.innerText = globalGain.toString();
gainSliderElement.addEventListener('input', () => {
  globalGain = +gainSliderElement.value;
  gainDisplayElement.innerText = globalGain.toString();
});

const chordIndices: {[chord:string]: number} = {
  'N.C.': 0,
  'C': 1,
  'C#': 2,
  'Db': 2,
  'D': 3,
  'D#': 4,
  'Eb': 4,
  'E': 5,
  'F': 6,
  'F#': 7,
  'Gb': 7,
  'G': 8,
  'G#': 9,
  'Ab': 9,
  'A': 10,
  'A#': 11,
  'Bb': 11,
  'B': 12,
  'Cm': 13,
  'C#m': 14,
  'Dbm': 14,
  'Dm': 15,
  'D#m': 16,
  'Ebm': 16,
  'Em': 17,
  'Fm': 18,
  'F#m': 19,
  'Gbm': 19,
  'Gm': 20,
  'G#m': 21,
  'Abm': 21,
  'Am': 22,
  'A#m': 23,
  'Bbm': 23,
  'Bm': 24,
  'C+': 25,
  'C#+': 26,
  'Db+': 26,
  'D+': 27,
  'D#+': 28,
  'Eb+': 28,
  'E+': 29,
  'F+': 30,
  'F#+': 31,
  'Gb+': 31,
  'G+': 32,
  'G#+': 33,
  'Ab+': 33,
  'A+': 34,
  'A#+': 35,
  'Bb+': 35,
  'B+': 36,
  'Co': 37,
  'C#o': 38,
  'Dbo': 38,
  'Do': 39,
  'D#o': 40,
  'Ebo': 40,
  'Eo': 41,
  'Fo': 42,
  'F#o': 43,
  'Gbo': 43,
  'Go': 44,
  'G#o': 45,
  'Abo': 45,
  'Ao': 46,
  'A#o': 47,
  'Bbo': 47,
  'Bo': 48
}

const chordPositions = ['1', '2', '3', '4', '5', '6', '7', '8']
const chordProgressionElements = chordPositions.map(
    i => document.getElementById('chord-' + i) as HTMLInputElement);

let preset1 = ['C', 'C', 'C', 'C', 'F', 'F', 'G', 'G'];
let preset2 = ['Am', 'Am', 'Am', 'Am', 'G', 'G', 'F', 'F'];

try {
  parseHash();
} catch (e) {
  // If we didn't successfully parse the hash, we can just use defaults.
  console.warn(e);
}

function parseHash() {
  if (!window.location.hash) {
    return;
  }
  const params = window.location.hash.substr(1).split('|');
  const chords = params[0].split(',');
  for (let i = 0; i < chordProgressionElements.length; i++) {
    chordProgressionElements[i].value = chords[i];
  }
  const preset1Values = params[1].split(',');
  for (let i = 0; i < preset1.length; i++) {
    preset1[i] = preset1Values[i];
  }
  const preset2Values = params[2].split(',');
  for (let i = 0; i < preset2.length; i++) {
    preset2[i] = preset2Values[i];
  }
}

function updateChordProgression(newChords: string[]) {
  for (let i = 0; i < newChords.length; i++) {
    const chord = newChords[i].toString();
    chordProgressionElements[i].value = chord;
    chordProgressionIndices.set(chordIndices[chord], i);
  }

  window.location.assign(
      '#' + newChords.join(',') + '|' +
      preset1.join(',') + '|' + preset2.join(','));

  resetRnn();
}

document.getElementById('c-am-dm-g').onclick = () => {
  updateChordProgression(['C', 'C', 'Am', 'Am', 'Dm', 'Dm', 'G', 'G']);
};

document.getElementById('c-bb-f-c').onclick = () => {
  updateChordProgression(['C', 'C', 'Bb', 'Bb', 'F', 'F', 'C', 'C']);
};

document.getElementById('dm-g-c').onclick = () => {
  updateChordProgression(['Dm', 'Dm', 'G', 'G', 'C', 'C', 'C', 'C']);
};

document.getElementById('am-g-f-e').onclick = () => {
  updateChordProgression(['Am', 'Am', 'G', 'G', 'F', 'F', 'E', 'E']);
};

document.getElementById('f-c-dm-bb').onclick = () => {
  updateChordProgression(['F', 'F', 'C', 'C', 'Dm', 'Dm', 'Bb', 'Bb']);
};

document.getElementById('dm-a').onclick = () => {
  updateChordProgression(['Dm', 'Dm', 'A', 'A', 'Dm', 'Dm', 'A', 'A']);
};

document.getElementById('reset-rnn').onclick = () => {
  const chords = chordProgressionElements.map((e) => {
    return e.value;
  });
  updateChordProgression(chords);
};

document.getElementById('preset-1').onclick = () => {
  updateChordProgression(preset1);
};

document.getElementById('preset-2').onclick = () => {
  updateChordProgression(preset2);
};

document.getElementById('save-1').onclick = () => {
  preset1 = chordProgressionElements.map((e) => {
    return e.value;
  });

  window.location.assign(
      '#' + preset1.join(',') + '|' +
      preset1.join(',') + '|' + preset2.join(','));
};

document.getElementById('save-2').onclick = () => {
  preset2 = chordProgressionElements.map((e) => {
    return e.value;
  });

  window.location.assign(
      '#' + preset2.join(',') + '|' +
      preset1.join(',') + '|' + preset2.join(','));
};

async function generateStep(loopId: number) {
  if (loopId < currentLoopId) {
    // Was part of an outdated generateStep() scheduled via setTimeout.
    return;
  }
  await math.scope(async (keep, track) => {
    const lstm1 =
        math.basicLSTMCell.bind(math, forgetBias, lstmKernel1, lstmBias1);
    //const lstm2 =
    //    math.basicLSTMCell.bind(math, forgetBias, lstmKernel2, lstmBias2);
    //const lstm3 =
    //    math.basicLSTMCell.bind(math, forgetBias, lstmKernel3, lstmBias3);

    c.map(val => {
      track(val);
    });
    h.map(val => {
      track(val);
    });
    const outputs: Scalar[] = [];
    // Generate some notes.
    for (let i = 0; i < STEPS_PER_GENERATE_CALL; i++) {
      // Use last sampled output as the next input.
      const eventInput = math.oneHot(lastSample.as1D(), EVENT_SIZE).as1D();

      const currentMinute = math.divide(currentStep, STEPS_PER_MINUTE);
      const currentBar = math.multiply(currentMinute, BARS_PER_MINUTE);

      // Dispose the last sample from the previous generate call, since we
      // kept it.
      if (i === 0) {
        lastSample.dispose();
      }

      const currentProg = math.divide(currentBar, CHORD_PROGRESSION_BARS);
      const progStartBar = math.multiply(math.floor(currentProg), CHORD_PROGRESSION_BARS);
      const currentChordBar = math.sub(currentBar, progStartBar);

      const currentChordPos = math.floor(math.multiply(currentChordBar, CHORDS_PER_BAR));
      const currentChordPosOneHot = math.oneHot(currentChordPos.as1D(), CHORD_PROGRESSION_SIZE.get()).as1D();
      const currentChordIndex = math.sum(math.multiply(currentChordPosOneHot, chordProgressionIndices));

      const barOffset = math.sub(currentBar, math.floor(currentBar));
      const quarterFloat = math.multiply(barOffset, QUARTERS_PER_BAR);
      const quarter = math.floor(quarterFloat);
      const quarterOffset = math.sub(quarterFloat, quarter);
      const division = math.floor(math.multiply(quarterOffset, DIVISIONS_PER_QUARTER));

      const chordEncoding = math.oneHot(currentChordIndex.as1D(), CHORD_ENCODING_SIZE).as1D();
      const quarterEncoding = math.oneHot(quarter.as1D(), QUARTERS_PER_BAR.get()).as1D();
      const divisionEncoding = math.oneHot(division.as1D(), DIVISIONS_PER_QUARTER.get()).as1D();
      const meterEncoding = math.concat1D(quarterEncoding, divisionEncoding);
      const conditioning = math.concat1D(chordEncoding, meterEncoding);

      const input = math.concat1D(conditioning, eventInput);
      //const inputProj = math.add(math.matMul(input.as2D(1, -1), inputProjWeights), inputProjBiases);

      //const output =
      //    math.multiRNNCell([lstm1, lstm2, lstm3], inputProj.as2D(1, -1), c, h);
      const output =
          math.multiRNNCell([lstm1], input.as2D(1, -1), c, h);
      c = output[0];
      h = output[1];

      //const outputH = h[2];
      const outputH = h[0];
      const weightedResult = math.matMul(outputH, fullyConnectedWeights);
      const logits = math.add(weightedResult, fullyConnectedBiases);

      const softmax = math.softmax(logits.as1D());
      const sampledOutput = math.multinomial(softmax, 1).asScalar();
      outputs.push(sampledOutput);
      keep(sampledOutput);
      lastSample = sampledOutput;

      const numSteps = math.sub(lastSample, SHIFT_OFFSET_LO);
      const isShift = math.multiply(math.clip(math.sub(lastSample, SHIFT_OFFSET_LO), 0, 1), math.clip(math.sub(SHIFT_OFFSET_HI, lastSample), 0, 1));
      const nextStep = math.add(currentStep, math.multiply(isShift, numSteps));
      if (i === 0) {
        currentStep.dispose();
      }
      keep(nextStep);
      currentStep = nextStep;
    }

    c.map(val => {
      keep(val);
    });
    h.map(val => {
      keep(val);
    });

    await outputs[outputs.length - 1].data();

    for (let i = 0; i < outputs.length; i++) {
      playOutput(await outputs[i].val());
    }

    // Pro-actively upload the last sample to the gpu again and keep it
    // for next time.
    lastSample.getTexture();
    currentStep.getTexture();

    if (piano.now() - currentPianoTimeSec > MAX_GENERATION_LAG_SECONDS) {
      console.warn(
          `Generation is ${
                           piano.now() - currentPianoTimeSec
                         } seconds behind, ` +
          `which is over ${MAX_NOTE_DURATION_SECONDS}. Resetting time!`);
      currentPianoTimeSec = piano.now();
    }
    const delta = Math.max(
        0, currentPianoTimeSec - piano.now() - GENERATION_BUFFER_SECONDS);
    setTimeout(() => generateStep(loopId), delta * 1000);
  });
}

let midi;
// tslint:disable-next-line:no-any
let activeMidiOutputDevice: any = null;
(async () => {
  const midiOutDropdownContainer =
      document.getElementById('midi-out-container');
  const midiInDropdownContainer = document.getElementById('midi-in-container');
  try {
    // tslint:disable-next-line:no-any
    const navigator: any = window.navigator;
    midi = await navigator.requestMIDIAccess();

    const midiOutDropdown =
        document.getElementById('midi-out') as HTMLSelectElement;
    const midiInDropdown =
        document.getElementById('midi-in') as HTMLSelectElement;

    let outputDeviceCount = 0;
    // tslint:disable-next-line:no-any
    const midiOutputDevices: any[] = [];
    // tslint:disable-next-line:no-any
    midi.outputs.forEach((output: any) => {
      console.log(`
          Output midi device [type: '${output.type}']
          id: ${output.id}
          manufacturer: ${output.manufacturer}
          name:${output.name}
          version: ${output.version}`);
      midiOutputDevices.push(output);

      const option = document.createElement('option');
      option.innerText = output.name;
      midiOutDropdown.appendChild(option);
      outputDeviceCount++;
    });

    midiOutDropdown.addEventListener('change', () => {
      activeMidiOutputDevice =
          midiOutputDevices[midiOutDropdown.selectedIndex - 1];
    });

    if (outputDeviceCount === 0) {
      midiOutDropdownContainer.innerText = MIDI_NO_OUTPUT_DEVICES_FOUND_MESSAGE;
    }

    let inputDeviceCount = 0;
    // tslint:disable-next-line:no-any
    const midiInputDevices: any[] = [];
    // tslint:disable-next-line:no-any
    midi.inputs.forEach((input: any) => {
      console.log(`
        Input midi device [type: '${input.type}']
        id: ${input.id}
        manufacturer: ${input.manufacturer}
        name:${input.name}
        version: ${input.version}`);
      midiInputDevices.push(input);

      const option = document.createElement('option');
      option.innerText = input.name;
      midiInDropdown.appendChild(option);
      inputDeviceCount++;
    });

    if (inputDeviceCount === 0) {
      midiInDropdownContainer.innerText = MIDI_NO_INPUT_DEVICES_FOUND_MESSAGE;
    }
  } catch (e) {
    midiOutDropdownContainer.innerText = MIDI_NO_OUTPUT_DEVICES_FOUND_MESSAGE;

    midi = null;
  }
})();

function timeToChordPos(t: number) {
  const bar = t * BARS_PER_MINUTE.get() / 60.0;
  const chord = bar * CHORDS_PER_BAR.get();
  return Math.floor(chord);
}

function playRoot(chordPos: number, startSec: number, endSec: number) {
  const chordIndex = chordProgressionIndices.get(chordPos % CHORD_PROGRESSION_SIZE.get());

  if (chordIndex > 0) {
    const rootPitchClass = (chordIndex - 1) % 12;
    const bassPitch = 36 + rootPitchClass;

    if (activeMidiOutputDevice != null) {
      try {
        activeMidiOutputDevice.send(
            [MIDI_EVENT_ON, bassPitch, BASS_VELOCITY], startSec);
        activeMidiOutputDevice.send(
            [MIDI_EVENT_OFF, bassPitch, BASS_VELOCITY], endSec);
      } catch (e) {
        console.log(
            'Error sending bass events to midi output device:');
        console.log(e);
      }
    }

    piano.keyDown(bassPitch, startSec, BASS_VELOCITY);
    piano.keyUp(bassPitch, endSec);
  }
}

/**
 * Decode the output index and play it on the piano and keyboardInterface.
 */
function playOutput(index: number) {
  let offset = 0;
  for (const eventRange of EVENT_RANGES) {
    const eventType = eventRange[0] as string;
    const minValue = eventRange[1] as number;
    const maxValue = eventRange[2] as number;
    if (offset <= index && index <= offset + maxValue - minValue) {
      if (eventType === 'note_on') {
        const noteNum = index - offset;
        setTimeout(() => {
          keyboardInterface.keyDown(noteNum);
          setTimeout(() => {
            keyboardInterface.keyUp(noteNum);
          }, 100);
        }, (currentPianoTimeSec - piano.now()) * 1000);
        activeNotes.set(noteNum, currentPianoTimeSec);

        if (activeMidiOutputDevice != null) {
          try {
            activeMidiOutputDevice.send(
                [
                  MIDI_EVENT_ON, noteNum,
                  Math.min(Math.floor(currentVelocity * globalGain), 127)
                ],
                Math.floor(1000 * currentPianoTimeSec) - pianoStartTimestampMs);
          } catch (e) {
            console.log(
                'Error sending midi note on event to midi output device:');
            console.log(e);
          }
        }

        return piano.keyDown(
            noteNum, currentPianoTimeSec, currentVelocity * globalGain / 100);
      } else if (eventType === 'note_off') {
        const noteNum = index - offset;

        const activeNoteEndTimeSec = activeNotes.get(noteNum);
        // If the note off event is generated for a note that hasn't been
        // pressed, just ignore it.
        if (activeNoteEndTimeSec == null) {
          return;
        }
        const timeSec =
            Math.max(currentPianoTimeSec, activeNoteEndTimeSec + .5);

        if (activeMidiOutputDevice != null) {
          activeMidiOutputDevice.send(
              [
                MIDI_EVENT_OFF, noteNum,
                Math.min(Math.floor(currentVelocity * globalGain), 127)
              ],
              Math.floor(timeSec * 1000) - pianoStartTimestampMs);
        }
        piano.keyUp(noteNum, timeSec);
        activeNotes.delete(noteNum);
        return;
      } else if (eventType === 'time_shift') {
        const shiftSec = (index - offset + 1) / STEPS_PER_SECOND;
        const timeOffsetSec = currentPianoTimeSec - pianoStartTimestampMs / 1000.0;
        const currentChordPos = timeToChordPos(timeOffsetSec);
        const nextChordPos = timeToChordPos(timeOffsetSec + shiftSec);
        if (nextChordPos > currentChordPos) {
          const seconds_per_chord = 60.0 / CHORDS_PER_BAR.get() / BARS_PER_MINUTE.get();
          const chordOffset = Math.floor(timeOffsetSec / seconds_per_chord);
          const startSec = chordOffset * seconds_per_chord;
          for (let pos = currentChordPos + 1; pos <= nextChordPos; pos++) {
            const offset = pos - currentChordPos;
            playRoot(pos, startSec + offset * seconds_per_chord, startSec + (offset + 1) * seconds_per_chord);
          }
        }
        currentPianoTimeSec += shiftSec;
        activeNotes.forEach((timeSec, noteNum) => {
          if (currentPianoTimeSec - timeSec > MAX_NOTE_DURATION_SECONDS) {
            console.info(
                `Note ${noteNum} has been active for ${
                                                       currentPianoTimeSec -
                                                       timeSec
                                                     }, ` +
                `seconds which is over ${MAX_NOTE_DURATION_SECONDS}, will ` +
                `release.`);
            if (activeMidiOutputDevice != null) {
              activeMidiOutputDevice.send([
                MIDI_EVENT_OFF, noteNum,
                Math.min(Math.floor(currentVelocity * globalGain), 127)
              ]);
            }
            piano.keyUp(noteNum, currentPianoTimeSec);
            activeNotes.delete(noteNum);
          }
        });
        return currentPianoTimeSec;
      } else if (eventType === 'velocity_change') {
        currentVelocity = (index - offset + 1) * Math.ceil(127 / VELOCITY_BINS);
        currentVelocity = currentVelocity / 127;
        return currentVelocity;
      } else {
        throw new Error('Could not decode eventType: ' + eventType);
      }
    }
    offset += maxValue - minValue + 1;
  }
  throw new Error(`Could not decode index: ${index}`);
}
