# DivInit case studies — diversity breaks naive collapse

Five examples from GAIA (Serper, Qwen3-8B) where Standard parallel fails
on all 4 threads (pass@4 = 0) while DivInit produces the correct answer
(pass@4 = 1). All exhibit **query collapse**: Standard threads issue
near-identical turn-1 queries, retrieve the same documents, and converge
on the same wrong answer.

Trajectories: `results/main_table_web_serper/qwen3-8b/gaia/`

---

## Case 1 — Horror movie metalepsis (gaia-49, seed 1)

**Question:** In Valentina Re's contribution to the 2017 book "World Building:
Transmedia, Fans, Industries", what horror movie does the author cite as having
popularized metalepsis between a dream world and reality? Use the complete name
with article if any.

**Gold:** A Nightmare on Elm Street

### Standard — 4/4 wrong

All four threads issue near-identical queries. The top search result is the
Academia.edu page for Re's chapter titled "The Monster at the End of This Book"
(about metalepsis in general). Every thread mistakes the chapter title for
the cited horror movie.

| Thread | Turn-1 query | Answer |
|---|---|---|
| 0 | Valentina Re 2017 book World Building horror movie metalepsis | The Monster at the End of This Book |
| 1 | Valentina Re 2017 book World Building horror movie metalepsis | The Monster at the End of This Book |
| 2 | Valentina Re metalepsis horror movie 2017 book | The Monster at the End of This Book |
| 3 | Valentina Re 2017 book World Building horror movie metalepsis | The Monster at the End of This Book |

### DivInit — 1/4 correct

| Thread | Seed query | Answer |
|---|---|---|
| 0 | Metalepsis in horror films as per Valentina Re's 2017 contribution | The Monster at the End of This Book |
| 1 | Which horror movie is cited by Valentina Re for dream reality metalepsis? | The Ring Two |
| 2 | Valentina Re's mention of a horror film with metalepsis between dream and reality | The Ring Two |
| 3 | Film cited by Valentina Re in 2017 book on metalepsis between dream and reality | **A Nightmare on Elm Street** |

Thread 3's framing emphasizing "dream and reality" retrieves different
results that surface the actual citation. Threads 0-2 are wrong but
*differently wrong* — the diversity ensures at least one thread escapes
the chapter-title trap.

---

## Case 2 — NASA award number chain (gaia-146, seed 3)

**Question:** On June 6, 2023, an article by Carolyn Collins Petersen was
published in Universe Today. This article mentions a team that produced a paper
about their observations, linked at the bottom of the article. Find this paper.
Under what NASA award number was the work performed by R. G. Arendt supported
by?

**Gold:** 80GSFC21M0002

### Standard — 4/4 wrong

All four threads pack every keyword into one query and retrieve the same
results. All converge on a different NASA award number from the wrong paper.

| Thread | Turn-1 query | Answer |
|---|---|---|
| 0 | Carolyn Collins Petersen June 6 2023 NASA award number R. G. Arendt | NNX15AF61G |
| 1 | Carolyn Collins Petersen June 6 2023 Universe Today R. G. Arendt NASA award number | NNX15AF61G |
| 2 | Carolyn Collins Petersen June 6 2023 Universe Today R. G. Arendt NASA award number | NNX15AF61G |
| 3 | Carolyn Collins Petersen June 6 2023 Universe Today R. G. Arendt NASA award number | NNX15AF61G |

### DivInit — 3/4 correct

The diverse queries decompose the multi-hop chain: some threads search for
the Universe Today article, others for the funding source directly.

| Thread | Seed query | Answer |
|---|---|---|
| 0 | Funding source for R. G. Arendt's research in Universe Today article June 6, 2023 | **80GSFC21M0002** |
| 1 | Paper linked in Universe Today article by Carolyn Collins Petersen from June 6, 2023 | 80NSSC22K0568 |
| 2 | NASA grant number associated with R. G. Arendt's work cited in Universe Today article June 6, 2023 | **80GSFC21M0002** |
| 3 | NASA award number for research cited in Universe Today article June 6, 2023 | **80GSFC21M0002** |

---

## Case 3 — Doctor Who script heading (gaia-22, seed 1)

**Question:** In Series 9, Episode 11 of Doctor Who, the Doctor is trapped
inside an ever-shifting maze. What is this location called in the official
script for the episode? Give the setting exactly as it appears in the first
scene heading.

**Gold:** THE CASTLE

### Standard — 4/4 wrong

All threads answer with the episode title "Heaven Sent" rather than the
in-script setting name.

| Thread | Turn-1 query | Answer |
|---|---|---|
| 0 | Doctor Who Series 9 Episode 11 location name | Heaven Sent |
| 1 | Doctor Who Series 9 Episode 11 maze location | Heaven Sent |
| 2 | Doctor Who Series 9 Episode 11 first scene heading | Heaven Sent |
| 3 | Doctor Who Series 9 Episode 11 maze location | Heaven Sent |

### DivInit — 1/4 correct

| Thread | Seed query | Answer |
|---|---|---|
| 0 | Official script name of the maze in Doctor Who Series 9 Episode 11 | Heaven Sent |
| 1 | First scene heading of Doctor Who S9E11 maze setting | Heaven Sent |
| 2 | Doctor Who S9E11 maze location as described in the script | Heaven Sent |
| 3 | Exact setting name in Doctor Who S9E11 maze episode | **THE CASTLE** |

Thread 3's phrasing "exact setting name" retrieves the actual script
rather than plot summaries that only reference the episode title.

---

## Case 4 — Equine veterinarian surname (gaia-96, seed 1)

**Question:** What is the surname of the equine veterinarian mentioned in 1.E
Exercises from the chemistry materials licensed by Marisa Alviar-Agnew & Henry
Agnew under the CK-12 license in LibreText's Introductory Chemistry?

**Gold:** Louvrier

### Standard — 4/4 wrong

All threads issue queries containing "Marisa Alviar-Agnew" and retrieve
pages about the textbook authors rather than the exercise content. Three
of four answer "Agnew" (confusing the textbook author for the veterinarian
mentioned in the exercise).

| Thread | Turn-1 query | Answer |
|---|---|---|
| 0 | equine veterinarian in 1.E Exercises chemistry materials Marisa Alviar-Agnew | Agnew |
| 1 | equine veterinarian in 1.E Exercises chemistry materials Marisa Alviar-Agnew | Agnew |
| 2 | equine veterinarian in 1.E Exercises chemistry materials by Marisa Alviar-Agnew | (could not determine) |
| 3 | equine veterinarian in 1.E Exercises chemistry materials Marisa Alviar-Agnew | Agnew |

### DivInit — 1/4 correct

The pool generates queries that approach from different directions. The
first seed drops the author names entirely and searches by platform.

| Thread | Seed query | Answer |
|---|---|---|
| 0 | CK-12 chemistry 08/21/2023 equine veterinarian | **Louvrier** |
| 1 | 1.E Exercises author surname in LibreText chemistry | Agnew |
| 2 | Henry Agnew's contribution to equine veterinary exercises | Agnew |
| 3 | Equine vet in chemistry materials by Marisa Alviar-Agnew | Keenan |

Thread 0's query drops the misleading author names and searches by the
CK-12 platform directly, retrieving the actual exercise page that names
the veterinarian Louvrier.

---

## Case 5 — Rooster and hamster song (gaia-93, seed 1)

**Question:** Who composed the song that was performed by a rooster and a
hamster in separate animated videos at separate tempos with different lyrics?
Answer using the format "First Last".

**Gold:** Roger Miller

### Standard — 4/4 wrong

Threads retrieve results about "Hampton the Hamster" (a viral internet
character) and never connect the rooster video to identify the original song.

| Thread | Turn-1 query | Answer |
|---|---|---|
| 0 | (no query issued) | (malformed) |
| 1 | song performed by rooster and hamster animated videos separate tempos | Hampton |
| 2 | song performed by rooster and hamster animated videos different tempos | Hampton The Hamster |
| 3 | song performed by rooster and hamster animated videos | Ray Stevens |

### DivInit — 2/4 correct

| Thread | Seed query | Answer |
|---|---|---|
| 0 | (approach 1) | **Roger Miller** |
| 1 | (approach 2) | Michael Stipe |
| 2 | (approach 3) | Walter Scharf |
| 3 | (approach 4) | **Roger Miller** |

Two threads independently identify Roger Miller as the composer. The diverse
queries approach the problem from enough angles that the connection between
the rooster video, the hamster video, and the original song ("Robin Hood"
whistle-stop) is made by multiple threads.

---

## Summary

| Case | Question topic | Standard | DivInit | Failure mode |
|---|---|---|---|---|
| 1 | Horror movie metalepsis | 0/4 (same wrong) | 1/4 | Chapter title mistaken for cited film |
| 2 | NASA award number | 0/4 (same wrong) | 3/4 | Wrong award from wrong paper |
| 3 | Doctor Who script heading | 0/4 (same wrong) | 1/4 | Episode title mistaken for script heading |
| 4 | Equine veterinarian | 0/4 (same wrong) | 1/4 | Textbook author mistaken for exercise character |
| 5 | Rooster/hamster song | 0/4 (all wrong) | 2/4 | Viral character mistaken for composer |

In every case, Standard threads collapse onto the same misleading retrieval
and answer identically wrong. DivInit's forced query diversity ensures at
least one thread phrases the question differently enough to escape the trap.
