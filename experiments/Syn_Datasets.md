# Synthetic Datasets

## General Rules
- Same **structure** as our dummy data.
- Don't forget the **line frequency (50 Hz)** as a major artifact.
- Alway 10 subjects
- fs always 1000Hz. Except BERA 25KHz, Alpha to save space possible 250Hz,
- Always same Channel position, use 10-20 system
- The trigger could be an issue because BIDS uses a different format with events, etc. For me, additional channels for the trigger—either analog or binary.
---

## Alpha
- Start with **eyes closed**, then switch **open/closed 3 times** (3× open, 3× closed, 30s each).
- Trigger is needed for closed/open see dummy data
- Generate **more realistic data**, e.g., including other EEG frequencies and artifacts like **blinking** (short bursts of lower-frequency activity with higher power than higher frequencies with lower power); generally more low-frequency content. Example: [DOI: 10.1109/BIOCAS.2019.8918700](https://doi.org/10.1109/BIOCAS.2019.8918700)  
- Use **32 channels** and a similar **topography** as in: [Norcia & Tyler, 2007](https://www.sciencedirect.com/science/article/pii/S1053811907011639?via%3Dihub) (they used an **average reference**).

---

## VEP
- Generate a **typical pattern-reversal VEP**: [link](https://link.springer.com/article/10.1007/s10633-016-9553-y)
- Trigger is needed for every reverse of the pattern see dummy data
- Use **32 channels** with **topography** similar to: [link](https://link.springer.com/article/10.1007/BF01131153), using **linked-ear reference**.

---

## SSVEP
- Stimulus: **pattern-reversal** same as in VEP.
- No Trigger
- Use **32 channels** with **topography** similar to: [Norcia & Tyler, 2007 PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6871301/pdf/HBM-28-323.pdf).

---

## P300 (preferably Oddball, more common)
- Stimulus: **acoustic oddball** paradigm (80% stndard stimulus and 20% oddball stimulus).
- one trigger for standard and another trigger for oddball. I used two channels for that
- Use **32 channels** with **topography** similar to: [Nature, 2023](https://www.nature.com/articles/s41598-023-27528-0).

---

## ASSR
- Use **32 channels** with **topography** similar to: [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6189392/).
- no trigger needed
- one erar or both ears for stimuli
- 40 Hz modulation × 1 kHz carrier
---

## BERA (preferably ABR, more common)
- Use **16 channels** with **topography** similar to: [Sciencedirect example 1](https://www.sciencedirect.com/science/article/pii/S0385814612800514?via%3Dihub) [PMC article example 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC6870971/)
- Trigger for every stimulus
- one erar or both ears for stimuli
