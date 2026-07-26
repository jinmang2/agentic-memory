"""Porter (1980) stemming algorithm — pure-stdlib, vendored.

Two callers need Porter stemming and both are reproducing an upstream that got
it from ``nltk``: the official snap-research/locomo ``normalize_answer``
(F1/BLEU-1, ``bench/locomo.py``) and ``rouge_score``'s ``use_stemmer=True``
tokenizer, which A-MAC's Confidence feature runs on
(``organizers/admission.py``). ``nltk`` is not a dependency here, so this is a
compact vendored implementation of the classic Porter (1980) algorithm rather
than a naive suffix-stripping fallback. Derived from Vivake Gupta's
public-domain Python port of Martin Porter's reference C implementation
(https://tartarus.org/martin/PorterStemmer/), lightly modernized.

It lives at the package root rather than under ``bench/`` because
``organizers`` may not import ``bench`` — ``bench.locomo`` imports
``agmem.memory``, which imports the organizers, so a cross-package import here
would close a cycle.

Only ``stem(word)`` is used; the class is kept self-contained and unit-tested
(tests/test_locomo_eval.py).
"""

from __future__ import annotations


class PorterStemmer:
    """Porter stemmer. ``stem(word)`` lowercases nothing — the caller is
    expected to pass an already-lowercased token (LoCoMo normalize lowercases
    first).

    ``mode`` selects step 1c's ``Y -> I`` condition, which is where
    ``nltk.stem.porter.PorterStemmer``'s default (``NLTK_EXTENSIONS``) deviates
    most often from Porter's 1980 paper:

    - ``"original"`` (default) — the paper's ``(*v*) Y -> I``: apply whenever the
      stem contains a vowel, so ``saturday -> saturdai`` and ``enjoy -> enjoi``.
    - ``"nltk"`` — ``(*c and not c) Y -> I``: apply only when the letter before
      the ``y`` is a consonant and the stem is longer than one character.
      ``happy -> happi``, ``cry -> cri``, but ``saturday`` and ``enjoy`` are
      left alone.

    **Both upstreams being reproduced here call nltk in its default mode** —
    snap-research/locomo's ``normalize_answer`` (our F1/BLEU-1) and
    ``rouge_score``'s ``use_stemmer=True`` tokenizer (A-MAC Confidence) — so
    ``"nltk"`` is the faithful choice and ``"original"`` is only the default to
    keep already-published run artifacts reproducible. The difference is not
    cosmetic: under ``"original"``, ``cry``/``cried`` do not conflate (``cry``
    stays ``cry`` while ``cried`` becomes ``cri``), changing token overlap.

    The mode choice was **measured to change nothing we have published**:
    re-scoring all 11,914 stored ``results/repro/*.records.jsonl`` QA pairs under
    both modes moves zero questions' F1 (``docs/research/amac-admission-gate.md``
    §4). So this is a closed risk, not an open decision — which is also why
    flipping the default is not worth a re-measurement.

    Step 1c is **not the only divergence**: over a 692-word LoCoMo vocabulary,
    ``"nltk"`` mode still differs from real nltk on 10 words via
    ``NLTK_EXTENSIONS`` features this port does not carry (its 16-entry
    irregular-forms pool: ``sky``, ``outings``…; a 2-character ``vc`` case added
    to the step-5a ``_ends_cvc`` test, which is why nltk keeps
    ``are``/``ate``/``use``; and a step-2 difference reached by ``emotionally``).
    Full nltk equivalence is a separate, low-priority task for the same reason."""

    def __init__(self, mode: str = "original") -> None:
        if mode not in ("nltk", "original"):
            raise ValueError(f"mode must be 'nltk' or 'original', got {mode!r}")
        self.mode = mode
        self.b = ""  # buffer for the word being stemmed
        self.k = 0
        self.j = 0

    def _cons(self, i: int) -> bool:
        """True if b[i] is a consonant."""
        ch = self.b[i]
        if ch in "aeiou":
            return False
        if ch == "y":
            return True if i == 0 else not self._cons(i - 1)
        return True

    def _m(self) -> int:
        """Measure: the number of consonant sequences between 0 and j."""
        n = 0
        i = 0
        while True:
            if i > self.j:
                return n
            if not self._cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowelinstem(self) -> bool:
        """True if 0..j contains a vowel."""
        for i in range(self.j + 1):
            if not self._cons(i):
                return True
        return False

    def _doublec(self, j: int) -> bool:
        """True if b[j-1..j] is a double consonant."""
        if j < 1:
            return False
        if self.b[j] != self.b[j - 1]:
            return False
        return self._cons(j)

    def _cvc(self, i: int) -> bool:
        """True if i-2,i-1,i is consonant-vowel-consonant and the second c is
        not w, x or y (used to restore a short e)."""
        if i < 2 or not self._cons(i) or self._cons(i - 1) or not self._cons(i - 2):
            return False
        return self.b[i] not in "wxy"

    def _ends(self, s: str) -> bool:
        """True if 0..k ends with s; sets j accordingly."""
        length = len(s)
        if length > self.k + 1:
            return False
        if self.b[self.k - length + 1 : self.k + 1] != s:
            return False
        self.j = self.k - length
        return True

    def _setto(self, s: str) -> None:
        """Set (j+1)..k to the characters in s."""
        self.b = self.b[: self.j + 1] + s
        self.k = len(self.b) - 1

    def _r(self, s: str) -> None:
        """_setto(s) if m() > 0."""
        if self._m() > 0:
            self._setto(s)

    def _step1ab(self) -> None:
        if self.b[self.k] == "s":
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._setto("i")
            elif self.b[self.k - 1] != "s":
                self.k -= 1
        if self._ends("eed"):
            if self._m() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowelinstem():
            self.k = self.j
            if self._ends("at"):
                self._setto("ate")
            elif self._ends("bl"):
                self._setto("ble")
            elif self._ends("iz"):
                self._setto("ize")
            elif self._doublec(self.k):
                self.k -= 1
                if self.b[self.k] in "lsz":
                    self.k += 1
            elif self._m() == 1 and self._cvc(self.k):
                self._setto("e")

    def _step1c(self) -> None:
        """Step 1c: ``Y -> I``, under whichever condition ``mode`` selects.

        The stem here is ``b[:k]`` (the word minus its trailing ``y``), so
        nltk's ``(*c and not c)`` test — "stem longer than one character and its
        last letter is a consonant" — is ``self.k > 1 and self._cons(self.k - 1)``.
        ``_cons``'s recursion for ``y`` only reads earlier indices, all of which
        lie inside the stem, so it matches nltk's ``_is_consonant(stem, ...)``."""
        if not self._ends("y"):
            return
        if self.mode == "nltk":
            applies = self.k > 1 and self._cons(self.k - 1)
        else:
            applies = self._vowelinstem()
        if applies:
            self.b = self.b[: self.k] + "i"

    def _step2(self) -> None:
        ch = self.b[self.k - 1] if self.k > 0 else ""
        if ch == "a":
            if self._ends("ational"):
                self._r("ate")
            elif self._ends("tional"):
                self._r("tion")
        elif ch == "c":
            if self._ends("enci"):
                self._r("ence")
            elif self._ends("anci"):
                self._r("ance")
        elif ch == "e":
            if self._ends("izer"):
                self._r("ize")
        elif ch == "l":
            if self._ends("bli"):
                self._r("ble")
            elif self._ends("alli"):
                self._r("al")
            elif self._ends("entli"):
                self._r("ent")
            elif self._ends("eli"):
                self._r("e")
            elif self._ends("ousli"):
                self._r("ous")
        elif ch == "o":
            if self._ends("ization"):
                self._r("ize")
            elif self._ends("ation"):
                self._r("ate")
            elif self._ends("ator"):
                self._r("ate")
        elif ch == "s":
            if self._ends("alism"):
                self._r("al")
            elif self._ends("iveness"):
                self._r("ive")
            elif self._ends("fulness"):
                self._r("ful")
            elif self._ends("ousness"):
                self._r("ous")
        elif ch == "t":
            if self._ends("aliti"):
                self._r("al")
            elif self._ends("iviti"):
                self._r("ive")
            elif self._ends("biliti"):
                self._r("ble")
        elif ch == "g":
            if self._ends("logi"):
                self._r("log")

    def _step3(self) -> None:
        ch = self.b[self.k]
        if ch == "e":
            if self._ends("icate"):
                self._r("ic")
            elif self._ends("ative"):
                self._r("")
            elif self._ends("alize"):
                self._r("al")
        elif ch == "i":
            if self._ends("iciti"):
                self._r("ic")
        elif ch == "l":
            if self._ends("ical"):
                self._r("ic")
            elif self._ends("ful"):
                self._r("")
        elif ch == "s":
            if self._ends("ness"):
                self._r("")

    def _step4(self) -> None:
        ch = self.b[self.k - 1] if self.k > 0 else ""
        if ch == "a":
            if not self._ends("al"):
                return
        elif ch == "c":
            if not (self._ends("ance") or self._ends("ence")):
                return
        elif ch == "e":
            if not self._ends("er"):
                return
        elif ch == "i":
            if not self._ends("ic"):
                return
        elif ch == "l":
            if not (self._ends("able") or self._ends("ible")):
                return
        elif ch == "n":
            if not (
                self._ends("ant") or self._ends("ement") or self._ends("ment") or self._ends("ent")
            ):
                return
        elif ch == "o":
            if not (self._ends("ion") and self.j >= 0 and self.b[self.j] in "st"):
                if not self._ends("ou"):
                    return
        elif ch == "s":
            if not self._ends("ism"):
                return
        elif ch == "t":
            if not (self._ends("ate") or self._ends("iti")):
                return
        elif ch == "u":
            if not self._ends("ous"):
                return
        elif ch == "v":
            if not self._ends("ive"):
                return
        elif ch == "z":
            if not self._ends("ize"):
                return
        else:
            return
        if self._m() > 1:
            self.k = self.j

    def _step5(self) -> None:
        self.j = self.k
        if self.b[self.k] == "e":
            a = self._m()
            if a > 1 or (a == 1 and not self._cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == "l" and self._doublec(self.k) and self._m() > 1:
            self.k -= 1

    def stem(self, word: str) -> str:
        """Return the Porter stem of ``word``. Words of length <= 2 are
        returned unchanged (Porter leaves them alone)."""
        if len(word) <= 2:
            return word
        self.b = word
        self.k = len(word) - 1
        self.j = 0
        self._step1ab()
        self._step1c()
        self._step2()
        self._step3()
        self._step4()
        self._step5()
        return self.b[: self.k + 1]
