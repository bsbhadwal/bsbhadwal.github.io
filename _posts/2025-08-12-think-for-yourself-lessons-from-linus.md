---
layout: post
title: Think for Yourself — Lessons from a Linus Torvalds Email
---

A few days ago, Linus Torvalds (yes, *that* Linus — the creator of Linux) sent a blunt rejection to a RISC-V patch submission. The email was, in classic Linus fashion, politically incorrect, gruff, and brutally honest.

And you know what? I agree with him.

Here’s why.

---

## 1. Mistakes Are Fine — Switching Off Your Brain Is Not

I have zero problem with getting things wrong. We all do it.  
What I *do* have a problem with is when someone stops thinking for themselves.  

If you accept a design choice or implementation detail just because “Bharat said so” (or your senior said so), you’re not growing. Blind obedience is the fastest way to become a mediocre developer.

**Your job is not to nod along.** Your job is to understand *why* something is being done, and if it doesn’t make sense, challenge it.

---

## 2. Question Authority — Always

The hierarchy in a project exists for coordination, not for intellectual surrender.  

Good seniors expect you to challenge them when something feels wrong. Great ones will thank you for it.  

When Linus calls out a pointless “helper” function, it’s not just about that code — it’s about the principle:  

- Don’t add complexity for no reason.  
- Don’t hide intent behind unnecessary abstractions.  
- Don’t let “this is how we’ve always done it” be the only justification.

---

## 3. Tough Feedback Makes You Better

Yes, Linus’ tone can be abrasive.  
Yes, you might feel stung if someone calls your code “garbage”.  

But here’s the thing: being pushed into difficult situations — and forced to defend your decisions — will make you a *much* better programmer.  

Mediocrity is comfortable. Growth is not.

---

### Final Thought

The real takeaway from that email isn’t “Linus is rude.”  
It’s this: **think for yourself, defend your choices, and never settle for mediocrity.**  

The world has enough developers who can follow instructions. What it needs more of are developers who can *think*.

---

> No. This is garbage and it came in too late. I asked for early pull
> requests because I'm traveling, and if you can't follow that rule, at
> least make the pull requests *good*.
>
> This adds various garbage that isn't RISC-V specific to generic header files.
>
> And by "garbage" I really mean it. This is stuff that nobody should
> ever send me, never mind late in a merge window.
>
> Like this crazy and pointless make_u32_from_two_u16() "helper".
>
> That thing makes the world actively a worse place to live. It's
> useless garbage that makes any user incomprehensible, and actively
> *WORSE* than not using that stupid "helper".
>
> If you write the code out as "(a << 16) + b", you know what it does
> and which is the high word. Maybe you need to add a cast to make sure
> that 'b' doesn't have high bits that pollutes the end result, so maybe
> it's not going to be exactly _pretty_, but it's not going to be wrong
> and incomprehensible either.
>
> In contrast, if you write make_u32_from_two_u16(a,b) you have not a
> f%^5ing clue what the word order is. IOW, you just made things
> *WORSE*, and you added that "helper" to a generic non-RISC-V file
> where people are apparently supposed to use it to make *other* code
> worse too.
>
> So no. Things like this need to get bent. It does not go into generic
> header files, and it damn well does not happen late in the merge
> window.
>
> You're on notice: no more late pull requests, and no more garbage
> outside the RISC-V tree.
>
> Now, I would *hope* there's no garbage inside the RISC-V parts, but
> that's your choice. But things in generic headers do not get polluted
> by crazy stuff. And sending a big pull request the day before the
> merge window closes in the hope that I'm too busy to care is not a
> winning strategy.
>
> So you get to try again in 6.18. EARLY in the that merge window. And
> without the garbage.

*Read Linus’ original email here: [lkml.org](https://lkml.org/) (search for “RISC-V Patches for the 6.17 Merge Window, Part 1”).*
