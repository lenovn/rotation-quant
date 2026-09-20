# Sparse paired D under frozen SP2

Parent k0nDFs PPL17.134821523344698. Identity under new SW rule PPL17.17204209888376, NLL2.8432826019598076; selected t=.25 PPL17.155105697678067, NLL2.8422958376250413. D net delta versus same-rule identity PPL-0.01693640120569384, NLL-0.000986764334766299. Versus parent deltaPPL=0.020284174333369265. Neither candidate beats parent. Do not promote.

Single channel1976 from first24train meanabs/weight-columnmax. Original96SA/16SP2alpha unchanged. Two full validations, 252728 predictions each. No gradients, added online transform, or new clipping. Parent retains its original112SW, while this experiment changes up SW by1/D and expands down SW when rowabsmax requires; identity control isolates that rule. FP64 paired equivalence and pack/reload/frozen checks passed at runtime; independent review pending.
