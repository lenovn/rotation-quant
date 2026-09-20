# Precision improvement loop

Latest user authorization: continuous evidence-driven experimentation; no GPU time budget. Original weights frozen, no distillation, no gradients, no online transform; single GPU memory and selection rules retained. Primary objective improve matched full validation; target BF16+1=14.6346577256. Strict W4A8 and mixed precision are reported separately.

## Round 1: weight/activation attribution

Hypothesis: layer1 weight restoration explains most of its mixed precision gain. Original C+SP2 PPL17.6423999759 is frozen historical control, no repeated baseline.

- layer1_w16: PPL=16.6678442769; NLL=2.8134813709; delta_NLL=-0.0568237235
- layer1_a16: PPL=17.4201334616; NLL=2.8576266328; delta_NLL=-0.0126784616
- layer1_down_w16a16: PPL=16.4325174479; NLL=2.7992621429; delta_NLL=-0.0710429514

Independent verifier PASS. Only down1 W16A16 gives16.43252, close to/slightly better than whole layer16.47254; focus next experiment on this projection. W-only dominates A-only conditionally; do not add contributions.

## Round 2: fixed-grid down1 rounding

Implement finite coordinate search on original C learnedSW, original SP2 alpha, original high-precision weights. Actual quantized inputs from32 train windows seed42;2048fit and2048disjoint-row heldout. Candidates0/32/128/512 steps, heldout MSE chooses one for full validation. No SA/SW/recalibration changes. Objective is local weight reconstruction under actual quantized input, not complete MLP reconstruction. Deployment still strict W4A8 at existing boundaries. Codefixed_grid_rounding.py and precision_loop.py; commands/logs in run directories.

## Round 2 result and Round 3 launch

# Down1 fixed-SW rounding result

Full validation PPL 17.395098280145866, NLL 2.8561884584957267; delta versus original C+SP2 -0.2473016957437828 PPL, -0.014116635883509865 NLL. Strict original W4A8 quantization scope retained, no extra high-precision module, no new scales/clipping/online transforms.

Selected 512 coordinate updates by disjoint-row train heldout MSE. Changed482403 weight codes in down1, original SW exactly retained. Selection sees actual frozen SP2 inputs; reference is original weight acting on these same inputs. This is weight-local error, not a total MLP loss. Original weights remain frozen. Pack/reload and full-model frozen checks passed. Independent verifier PASS.

Next hypothesis: aggregate down-family rounding may retain benefit. Run ../down-d-search-20260913.6qBcoE via PRECISION_LOOP=rounding_downs bash scripts/phase2/46_run_down_d_search_local.sh; propagate selected preceding W4A8 layers, retain original112SW and96SA/16SP2 alpha, choose each layer among original and32/128/512 update candidates.

## Round 3 result and Round 4

# Sequential down-family rounding

Full validation PPL=17.546664783380706, NLL=2.8648638910188193; delta_NLL vs original C+SP2=-0.005441203360417202. All16 layer local heldout objectives selected512; original C SW/SA/SP2 alpha frozen. Sequential upstream selected weights are propagated to next layer, independent candidates include original codes. Independent verifier PASS.

Result is worse than single down1 rounding17.3950982801. Do not promote family over current best. Local weight reconstruction improvements do not necessarily accumulate in end-to-end NLL.

Next experiment: same down1 scope and candidate budget as GXOlZS, but use same-input fixed-R unquantized BF16 MLP reference, including gate/up and activation error in reference mismatch. Candidate local matmul remains FP32 with BF16 dequantized weights/inputs, so this is a proxy; final validation uses actual BF16 GEMMs. No teacher model/KL, no gradients, no changed scales/clipping. Run ../down-d-search-20260913.v2WHxW; command PRECISION_LOOP=rounding_mlp bash scripts/phase2/46_run_down_d_search_local.sh.

## Round 4 result and Round 5

# Same-input MLP reference rounding

Full validation PPL=17.40639276660993, NLL=2.856837539125371, delta_NLL=-0.013467555253865449. Better than original17.64239998 but not better than weight-only down1 rounding17.39509828; retain prior best. Reference includes BF16 gate/up/down at same actual normalized MLP input and same R. Candidate local objective is FP32 matmul with BF16 quantized weights/input; final evaluation uses actual BF16 forward. Original SW/SA/SP2 alpha fixed, no gradient or teacher KL.

Next hypothesis: 2048 calibration rows may permit spurious input correlations and row-heldout from same windows may be too weak. Keep same32train windows but fit full24windows (49152rows), select on remaining8fullwindows (16384rows). Compare original and512/2048/8192 coordinate steps. Fixed-SW weight-only objective retained to match current best. Run ../down-d-search-20260913.sWJwrk via PRECISION_LOOP=rounding_coverage bash scripts/phase2/46_run_down_d_search_local.sh.

## Round 5 result and Round 6

# Full-window down1 rounding

Validation PPL=17.156840423438286, NLL=2.8423969525985595; delta_PPL=-0.4855595524513632, delta_NLL=-0.02790814178067702. New best original W4A8 scope, original SW/SA/SP2 alpha retained. Independent verifier PASS; pack and recorded code count4765850 match. Selected8192 steps by8full train heldout windows (16384rows), fit24full windows (49152rows), all drawn from original32-window seed42 settings. This changes both data coverage and search depth relative to GXOlZS, so benefit cannot be assigned to either alone.

Next round: 16down sequential extension with heldout true NLL acceptance instead of relying solely on MSE. Per layer fit24/hold8, candidate0/512/2048/8192. MSE prefilters; each surviving candidate scored with8-window true next-token NLL; no gain retains original current-layer codes. Previous selected prefix's NLL is reused as exact current-layer identity control. Final fullvalidation only after all layers. Run ../down-d-search-20260913.k0nDFs with PRECISION_LOOP=rounding_downs_nll bash scripts/phase2/46_run_down_d_search_local.sh. Prior best saved independently, not overwritten.

## Round 6 result and Round 7

Sequential train-NLL acceptance selected down layers1/4/8 at8192 steps,10 at2048,11 at512; other11 retain original codes. Full validation PPL17.134821523344698, NLL2.8411127393601405, deltaPPL=-0.507578452544952, deltaNLL=-0.029192355019096. Independent verifier PASS; 49train scores and1fullvalidation. Original112SW/96SA/16SP2alpha unchanged. Best strict W4A8 so far; target gap2.500163797701495PPL.

Round7 hypothesis: local fixed-grid rounding gain is limited; selective channel scaling can improve salient W4 columns while SP2 handles the rescaled activation. Parent k0nDFs. Train chooses one layer1 channel by meanabs pre-SP2 activation/down columnmax over24 fullwindows. D strengths0/.25/.5/.75/1; geometricmean1 and bounds[.25,4]. Pair up/D and down*D from unquantized same-R weights. upSW=C/D; downSW=max(CSW,current rowabsmax/7). Original SP2 alpha never shrinks or changes. Same-rule D=I isolates changed SW and rerounding. Fit24windows/hold8, rounded candidate selection then true8window NLL; at most2fullvalidation. No gradients/distillation/online ops. Commands and live logs: ../down-d-search-20260913.HymqCR/command.txt and experiment.log. PythonPID131716 GPU1.

## Round7 result and Round8

Actual completed run iyIuSM (HymqCR and8AzYuY interrupted before results). Identity SW rule17.17204209888376; selectedt.25 D17.155105697678067. D net gain-.0169364012 but parent17.134821523344698 remains better. Do not promote or enlarge D strength. Next: frozen parent residual diagnostics allA16/downA16/downW16/non-downW16, four fullvalidation conditions, no recalibration; use conditional NLL gaps to select next weight family.

## Round8 result and Round9

Residual diagnostics ZUk3sH complete, independent PASS. Frozen parent17.134821523344698. AllA16 PPL15.863821573150512/NLL2.764041143874271; downA16 15.922558633306572/NLL2.7677368856749784; downW16 15.77839572928717/NLL2.758641645438812; non-downW16 16.54871401857111/NLL2.8063083959986814. No recalibration, all parentcodes/scales shared unless specific W/A restored. These conditional gains are not additive; all are diagnostics not deployable strict W4A8 improvements.

Round9 hypothesis: remaining down activation gap and changed rounded weights justify adapting SP2 alpha using actual train NLL rather than original localMSE selection. Parentk0nDFs W4codes/all112SW/96non-downSA stayfrozen. Each of16down alpha may multiply original by1/1.125/1.25/1.5/2; never shrink threshold.8last trainwindows fromsame32seed42; selectgreedily eachlayer withacceptedprefix, retainIifnoNLLgain.64train scores, atmost1fullvalidation. Fixedinput saturationthreshold doesnotdecrease, but changedupstream canchangeactual downstreamsaturationcount. No gradient,distillation,online transform. RunVVciEd via PRECISION_LOOP=sp2_alpha_expand LOOP_PARENT=...k0nDFs bash scripts/phase2/46_run_down_d_search_local.sh.

## Round9 result and Round10

VVciEd complete: fullvalidationPPL16.701812082653035,NLL2.815517221479444, deltaPPL vs k0nDFs=-.4330094406916629 and originalC=-.9405878932366143. Accept alpha factors layer0=2,layer2=1.5,layer3=1.25,layer13=1.25; other12 unchanged. All112SW andparentintegerweights frozen.64eightwindowtrain scores;1fullvalidation. Independentresultreviewpending.

Next hypothesis: layer0 chosenatupperfactor2 while otherlayersnotatboundary; searchonlylayer0 currentalpha times1/2/4/8/16, max4newtrain scores+1validation. Other15selectedalpha fixed. Existingbaseline currentVVciEd. No newclippingthresholdshrink, unchangedSP2codebook/weights/SW/96SA.

## Round10 result and Round11

89KzK6 boundarysearch complete. Onlylayer0alpha searched relativeVVciEd by1/2/4/8/16. Selected8x current=16x original=25.76784531118807, next16x worse ontrain. FullPPL16.508818358818367,NLL2.803894684130134, deltaPPL versusVVciEd=-.19299372383466817, versusoriginalC=-1.1335816170712825. FourtrainNLLscores plusonefullvalidation. Stop enlargingalpha.

Round11 hypothesis: floor/ceil restriction prevents correlatedfeatures compensating for fixed-grid weightquantization. Start exactly k0nDFs down1 codes, keep89KzK6 alpha andall112SW/96SA. Onlydown1 neighbor±1 code updates inside[-8,7], no changes tobaseFPweights.24fulltrainfit/8hold; candidates0/512/2048/8192. Analyticquadraticcost withactualBF16 dequantizedweights, actualSP2 quantizedinputs afterselectedupstream. LocalMSEprefilterthen8windowtrueNLL chooses;0retainsparent. Maxonefullvalidation. CPUtest proves compensationtoy inaccessibletofloorceil butreachablebyneighbor, originaltestsretainedPASS.

## Round11 result and Round12 matched control

Neighbor9Zycd9 complete independentPASS: PPL16.161902488363097,NLL2.7826567744038146. Train selects8192steps, changes2898198down1codes fromparent. All112SW/96SA/16SP2alpha unchanged, other15downcodes unchanged. DeltaPPL=-.3469158704552697 vs89 and-1.4804974875265522 vsoriginalC.

Matched floorceil continuationFwSJv9 fromexact89 parent, samefit/hold data, samebudgetcandidates: selected512, fullPPL16.491503247836636,NLL2.802845293595272. Neighbor net versusmatchedcontrol=-.329600759473539PPL,-.0201885191914574NLL. Supports neighbor-code freedom, not onlymoreoptimization.

Round13: sequentialextensionall16down starting9Zparent (includingallselectedalpha), fixedSW,24fit/8heldout,0/512/2048/8192 neighbor candidates. Eachlayer identityusesparentcurrentweight and acceptedprefix; trueNLLacceptance, explicitlyrestorebestthenpropagate.49trainNLLscoresmaximum, atmost1fullvalidation; unchangedallparent model reusespriorfullvalidation. No teacher/gradients/newonlineops/thresholdshrink.

## Round13 result and Round14/15 planned matched pair

Ds4vSf complete: fullPPL16.122226577621436,NLL2.7801988527027004, deltaPPLvs9Z=-.03967591074166066, vsoriginalC=-1.5201733982682128. StrictW4A8alloriginal112SW/currentalpha retained. Runtime961.996s, allocator5.91796875GiB, onefullvalidation. Gainslimiteddespiteseveralacceptedtrainchanges.

NextmatchedpairfromsameDs4vSfparent: down1 neighborcodes0/512/2048/8192,24fit/8hold, actualW4gate/up+staticSP2inputs. Maincase targets sameactualnormalizedMLP input fedthrough same-R unquantizedBF16gate/up/down; weight-localcontrol targetsfpdown actingonsamequantizedinput. Onlytargetchanges. No teacherlogits/KL, noautograd, noSW/alphachanges. CandidateMSE isFP32localproxy; actualtrainNLLchooses, fullvalidationjudges. IndependentCPUalignment/arbitrarytargetupdatePASS.

## Round14 rejected by train NLL

YocmPn complete83.93s, nofullvalidationbecauseidentityselected. CompleteMLP reference MSEdrops .00153465->.00096107 fit, .00096439->.00072018 heldout at8192, buttrainNLL parent2.73017928 versus512=2.74041650,2048=2.74343299,8192=2.76074262. Allnonidentityrejected. PersistedmodelidenticalDsparent andpriorPPL reusedexplicitly. LocalMLPproxyimprovementdidnottranslateintotaskNLL. Proceedmatchedweight-localcontinuationbeforeexternalC4check.

## Round15 matched continuation rejected

i5BuzT weight-localneighbor continuation fromDs4vSf alsoselected0. ParenttrainNLL2.7301792795 versus5122.7360206186,20482.7340562535,81922.7337067540. LocalMSEslightlyimprovedbutNLLworse; savedmodelidenticalparent, nofullvalidation. Thusneitheradditionalweight-localstepsnorcompleteMLPreferencebenefitscurrentdown1 onthisfixedgrid. Bestremains16.122226577621436.

Next externalcheck: freezeDs4vSf, evaluateoriginalC4savedinput_tokens.pt1024windows2048, chunks128,2096128predictions. Comparehistoricalsame-token BF1621.843397624525274 andoriginalCSP229.539535457544623. No C4calibration/selection, no newmodel/format/precisionexceptions.

## External C4 result (not used for selection)

EH3Kju completed frozenDs4vSf onexisting1024C4windows: PPL27.062581939332095,NLL3.2981520335394494; oldCSP229.539535457544623/NLL3.385729551100929. DeltaPPL=-2.476953518212529, deltaNLL=-.08757751756147947. BF16historical21.843397624525274, so5.219184314806821PPLgapremains.2096128targets,same8segments/metadata,no calibration/selection. Independentartifactreviewpending.

ReturntoWiki2: train-only48non-downW16restorationdiagnostics(16layers*gate_up/qkv/o), allA8/currentalpha frozen; sameparentDS everycase. Selectatmost2positivegroupsbyconditionaltrainNLL forfiniteW4codeoptimization. NotusingC4scoreforranking; highprecisioncasesarelocalizationonly.

## Non-down sensitivity and bounded W4 pilot

GpmBQJ completed 48 independent train-only restoration cases, no full validation. Top groups layer13 gate/up trainNLL2.7252634132005347 and layer14 gate/up2.726920610579061 versus parent2.730179279517435. This is conditional localization, not final PPL attribution.

JVI83z launched GPU6 PythonPID490569: PRECISION_LOOP=non_down_rounding LOOP_PARENT=/home/dongpeiyan/projects/rotation-quant/runs/phase2/down-d-search-20260913.GpmBQJ bash scripts/phase2/46_run_down_d_search_local.sh. Four gate/up matrices only; same SW, down codes and alpha; neighbor steps0/512/2048/8192, current-prefix train selection, at most one complete validation. Independent six-target CPU simulation readiness PASS. C4 external frozen check independent PASS; not used for ranking.

## Four-matrix pilot result

JVI83z exited0, one complete Wiki2 validation252728targets: PPL16.11180256293439/NLL2.779552081862793, delta parentPPL=-.010424014687046679/NLL=-.0006467708399076066. FinaltrainNLL2.727271304469372. This is a small non-down W4 improvement, not evidence that all96 should be changed blindly. Independent artifact review pending; GPU job finished. Continue evidence-driven loop; BF16+1 target not achieved. C4 remains external frozen evidence only.

JVI83z independent artifact PASS: original16down/112SW/96SA/alpha preserved, 3 non-down matrices changed (13gate retainedI). Promote16.11180256293439. Next diagnostic reads all20 parent records and actual parent SP2alpha; reuse fullA8, measure allA16/downA16/downW16/non-downW16 without recalibration. Earlier ZUk3sH preceded alpha and neighbor-code improvements, so cannot describe current residual bottleneck.

BJ3lSl GPU6 launcher529407 command: PRECISION_LOOP=residual_diagnostics LOOP_PARENT=/home/dongpeiyan/projects/rotation-quant/runs/phase2/down-d-search-20260913.JVI83z bash scripts/phase2/46_run_down_d_search_local.sh. CPU/source independent readiness PASS with112 wrappers20 replacements; no duplicate process at launch.

BJ3lSl completed4 frozen diagnostics: allA16=15.501948728965315, downA16=15.553530437314299, downW16=15.237642606409151, non-downW16=15.067222135354244. FullA8=16.11180256293439 reused. All252728targets; no recalibration. Conditional restoration gains not additive. Next five-matrix proposal /home/dongpeiyan/projects/rotation-quant/runs/phase2/non-down-second-batch-20260913.99l3b8_l uses prior GpmBQJ ranking (explicitly not remeasured on JVI), excluding tested13/14gate_up, chooses11gate_up/15qkv. Parent JVI; unchanged112SW/96SA/16alpha and accepted20records;0/512/2048/8192, real current-prefix trainNLL acceptance and at most1fullvalidation.

BJ3lSl independent final PASS. Second batch rectangular solver/pack readiness PASS; qvjyDe launchedGPU6 launcher547483: PRECISION_LOOP=non_down_rounding LOOP_PARENT=/home/dongpeiyan/projects/rotation-quant/runs/phase2/non-down-second-batch-20260913.99l3b8_l bash scripts/phase2/46_run_down_d_search_local.sh. Expected25records preservingparent20.

Second batch qvjyDe exited0: PPL16.11105493245762/NLL2.7795056781273693, delta JVI PPL=-.00074763047676818/NLL=-.00004640373542353. Onlylayer11gate512 accepted;11up and15Q/K/V retainI. LocalMSE reduction fails trainNLL for allQKVcandidates. This tiny deterministic score gain does not establish meaningful statistical improvement; stop broadening this local non-down approach now. Independent artifact review pending.

Next bounded hypothesis (not yet implemented): previous sparseD selected channel1976 by meanabs; RMS may identify different rare/high-energy channel relevant to quadratic W4 error. Selectonechannel fromactualpreSP2 trainfirst24windows RMS dividedbydownFPcolmax, no hardcoded1417. Five strengths0/.25/.5/.75/1, positiveGM1[.25,4], independentpairedFPup/down fusion, noonlineops. KeepcurrentSP2alpha and96SA. upSWtransport C/D, downSWmax(C,transformedBF16rowabsmax/7), explicitI+sameSW/solvercontrol versusD; candidate neighborINT4solve fromfreshFP eachtime, samefinite steps, actualtrainNLLselection. Preserveall currentnon-down records. Maximum2fullvalidation(Irule andselectedD), originalparent remainscontrol. Do not claim RMS ranks1417 before measurement or reuse old16-only loader unmodified.

qvjyDe independent artifact PASS:25records, allparent20 unchanged, allSW/alpha unchanged,1fullvalidation252728targets, deltaPPL=-.000747630476769956. Numerically best16.11105493245762; goal14.634657725643203 gap1.4763972068144167. No activeGPUjob; goal remainsactive andnextRMS-D experiment queued.

User correction: old68/31 attribution belongs originalC, notcurrentbest. RMS-D alreadyusesqvjyDe16.11105 asparent. BJ3lSl diagnostics belongprecedingJVI16.11180, so runexactlatest4frozenconditions first. EZPTam GPU6 launcher590003: PRECISION_LOOP=residual_diagnostics LOOP_PARENT=/home/dongpeiyan/projects/rotation-quant/runs/phase2/down-d-search-20260913.qvjyDe bash scripts/phase2/46_run_down_d_search_local.sh. CPU tests test_down_d_search/test_fixed_grid_rounding10PASS. RMS-D implementationpendingindependentreview, noD GPUstarted yet.

Exact latestqvjyDe diagnostic EZPTam exited0: {"full_a8": {"ppl": 16.11105493245762, "nll": 2.7795056781273693}, "all_a16": {"ppl": 15.49485525361626, "nll": 2.7405080497063254}, "down_a16": {"ppl": 15.548379865182714, "nll": 2.7439564444629894}, "down_w16": {"ppl": 15.23049418236542, "nll": 2.723299614336354}, "non_down_w16": {"ppl": 15.067222135354244, "nll": 2.7125216648812587}}. OriginalC68/31 not applicable. Independent final review pending.

RMS-D readiness independentPASS5D*4trials CPU, outputs26records preservingother24; source3filesmodified, CPU10testsPASS. 7U1l7n launchedGPU1 launcher602172: PRECISION_LOOP=sparse_d_rms_sp2 LOOP_PARENT=/home/dongpeiyan/projects/rotation-quant/runs/phase2/down-d-search-20260913.qvjyDe bash scripts/phase2/46_run_down_d_search_local.sh. No hardcoded salientchannel; measuredtrainRMSdirection. Parentalpha frozen, SWup=C/D, SWdown=max(C,newfullrange). CandidatefreshFP->neighborINT4steps0/512/2048/8192, true8trainNLL chooses; max2fullvalidation.

User steering: asks whether fixed-C optimization has plateaued and originalC construction is bottleneck; do not equate tiny local gains with globalC limit. Live C/logs/rotation.log229 and launcher verify originalC trained17R/96SA/112SW with downA16, W4-aware100steps; finaldownSP2A8 wasnot jointlyseen. Hypotheses distinguish numericaltransform damage, R/SW/SA choice andforward mismatch, localMSE proxy limitations. Add c_integrity mode toprecision_loop/down_d_search: restoreall112FPweights/112A16 under C-R/fusion, onefullvalidationvsoriginalunrotatedBF16. CPUindependent112wrapperactualforward/quantbypass/boundaryPASS; noRretrainingauthorizedorperformed. Queuedafter currentRMSrun.

7U1l7n complete independentPASS:channel1417 RMS,t1,selectedneighbor8192,train2.725912757341436. I sameSWrulePPL16.118253028142952/NLL2.779952358261725;Dselected16.05715342622255/NLL2.7761544466207675. D vsparent=-.05390150623507PPL; vsI=-.06109960192040. DGM1/min.999830842/max3.999322891.26records,other24unchanged,originalalphasunchanged,2fullvalidations. Promote newbest; 14.63466goalstillnotachieved. Important: newartifact selected_d/packed_model.pt has weights26 andD; up1/down1SWdifferentfromC. Anyfuture loader must preserveallrecords/scales andconstruct sameD-fusedFP references; olderrawrounding16-only/allSW=Cloaders invalid. This result contradicts claim that allC-anchored local improvement exhausted, butgainmodest.

Ie8eEI launchedGPU1 launcher631633: PRECISION_LOOP=c_integrity bash scripts/phase2/46_run_down_d_search_local.sh. IndependentreadinessPASS; originalC-R/FPfusionallW16A16 check, noDartifact applied. Distinguish C numericalpath fromquantizationparameterchoice.

Ie8eEI exited0,1fullvalidation252728targets: C-R/fusion allW16A16 PPL13.647488162677584,NLL2.6135554873114346 versus originalBF1613.634657725643203/2.612614913352714. Delta+.012830437034381248PPL/+.000940573958720492NLL. No evidence of multi-point performance destruction by the fullprecision transform path; aggregatePPL doesnotprovepointwiseequivalence oroptimalR. C originaltrainingdownA16 vsfinalSP2A8 mismatchverified rotation.log229. LocalMSE failure/NLLplateau cannotestablishCgloballimit; newD16.05715 modestimprovement iscounterevidence toexhaustion. Priorityshift tocontrolledquantizationparameter/forward-mismatch comparisons; noauthorizationinferredforgradienttrainingorunfreezingR/96SA. Independentfinalreviewpending.

Ie8eEI independent final PASS; originalBF16matchedreference reread verified. Currentbest16.05715342622255;Cfullprecision13.647488162677584;difference2.409665263544964PPL/.16259895930933288NLL. AllGPUjobs completed thisturn; goalactive.

Next controlled SW hypothesis: integercode search plateau maypartlyreflectfixedSW grid. New fixed_code_sw mode derives16down rowLS directions fromactualfrozen7U1l7nselectedD parentSP2inputs first24trainwindows, sameD-fusedFP downreference. Freezeallcodes/R/D/96SA/16alpha/non-downSW. Constrain SW onlyexpand1..1.125current, globalbeta0/.25/.5/.75/1,4true8windowtrainNLLscores, atmost1fullvalidation;I reusesverifiedparent. FittingparentpathfixedwhileeachcandidateNLLactualnewforward. No gradient/teacher/newonlineops/thresholdshrink. Testscheckconstrainedoutputerroroptimum,zeroobservedrow,positivity/cap/NaN; independentreadinesspending. This tests boundedexpansion,notallCscalechoices orRoptimality.

cohqXT launchedGPU1 launcher659428: PRECISION_LOOP=fixed_code_sw LOOP_PARENT=/home/dongpeiyan/projects/rotation-quant/runs/phase2/down-d-search-20260913.7U1l7n bash scripts/phase2/46_run_down_d_search_local.sh. IndependentCPUreadinessPASS16parentcaptures/4candidateconfigures/26records/packreload;2newCPUtestsPASS.

cohqXT exited0,independentartifactPASS. beta0trainNLL2.725912757341436, .25=2.7308574454394403,.5=2.7307090951493924,.75=2.730460386578702,1=2.7309014428858553. SelectedI;26recordsallcodes/SW/D/alphaexactparent;0newfullvalidation,16.05715342622255explicitlyreused. Reject onlythisboundedSWexpansiondirection.

Proposed next experiment, NOT authorized or implemented: R-only gradient adaptation paired forward-mismatch test. Preserve bestD16.05715 unchanged. Botharms start same originalC R and112learnedSW/96SA, D=I forcleanC-construction comparison; freezeallFPweights/SW/SA and SP2alpha(currentexpandedalphas), trainonlyR1/R2. SameWiki2trainloader/seed42/2048/globalbatch8/100steps perarm, sameoptimizer/LR andRTN[-8,7]forward (finiteoptimizerdetails reviewed before launch). ArmA downA16 training;ArmB frozenSP2A8 training withsameexactforward andSTEbackward only. Non-downINT8 both. FinalbothstrictW4A8samealpha/RTN; atmost3completevalidation(startplus2ends),singleGPU12GiB. No teacher/distillation/GPTQ/extraonlineops/scalelearning/thresholdsearch. Requires userexplicitlyrelaxno-gradient andfrozenC-R only;96SA/SW remainfrozen. No suchtrainingrunstarted.

Concrete proposed optimizer for bothR-only arms: existingSGDG stiefelTrue, RLR0.15 (one tenth historicalC LR tolimitcontinuationperturbation), cosine, warmup10, weightdecay0, BF16/gradientcheckpointing, singleGPUbatch1 accumulation8, seed/data_seed42,100updates each. Proposal explicitlyawaitinguser scopeclarification viaasyncquestion; elapsedtime isnotapproval. Continue to preserve no-gradient/frozenR until an actual affirmative response.

Read-only R-only feasibility audit completed; independentreview agrees currentflags cannotfreezeSW/SA. Identifiedguard/initial/periodic/final refreshanddownA16helper interactions; hardSP2 lacksusefultraininggradient. Concreteproposal r_only_pair_proposal.json recordsdata/RTNstart/SGDG0.15/100stepsx2/resources/requiredchanges andchecks. No permissionresponse received,no training/sourcechanges. Originalbest16.05715 untouched. Previousgoalturnprogress;thisauditprogresschangesimplementationplan.

Blocked audit: scope boundary persisted across3consecutivegoalturns (SWresult+request; sourceaudit+concreteproposal; currentlivecheck). No affirmativepermissionresponse,no optimize_rotation/down_d_search process, preservedbestartifactexists16.05715342622255;goal14.634657725643203 remainsunmet by1.4224957005793453. SelectedC-construction comparison cannotlaunchwithinexistingno-gradient/frozenR restrictions; preparationfinished. Markautomaticgoalblockedpendingexplicituserdecision,notcomplete; thisdoesnotclaimallpossiblegradientfreealgorithmsmathematicallyexhausted.
