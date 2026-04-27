# Spectro 

a tiny python tool to generate spectrograms for audio files and conduct heuristic "lossy source" analysis. 

## What can this be used for

I needed a way to quickly spot resampled or lossy-upconverted audio files. So I built this to:
* Render spectrograms quickly.
* Detect frequency shelf bands and cut-off points.
* Compare two audio files' spectra side by side.
  
## WARNING!

 The results are a hypothesis based on signal processing, not definitive proof of a track's provenance, sample rate, or codec.


## How to use

```
spectro "song.flac"                # basic spectrogram
spectro "song.flac" --detect       # lossy source heuristics
```

**For more details see [MANUAL.md](MANUAL.md).**



## An important remark on detection results

- A "Closest cutoff resemblance" result does not imply that the analyzed audio file meets the criteria of that particular profile. It only tells you which lossy profile's cutoff range looks similar.

- The spectral verdict (PASS / WARN / FAIL / INCONCLUSIVE) does not prove losslessness or lossiness. PASS simply means no obvious lossy signature was found; FAIL means strong evidence of lossy transcoding or upsampling. 

- Lack of ultrasonic energy is normal for archival audio content, so do not immediately assume that your track has been upconverted if it lacks ultrasonic frequencies.


## More Information

### Why?

No idea, I knew that C++ would be faster. But, I don’t know C++ well enough to build this in it yet. This was just a fun way to learn more about the math behind audio analysis & more.

### Methodology

**All** code was written by me, drawing from existing research but implemented independently. No AI was used.

## License
This software is licensed under the **MIT License**.