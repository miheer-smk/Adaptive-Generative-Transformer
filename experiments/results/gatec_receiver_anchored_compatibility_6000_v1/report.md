# Receiver-anchored compatibility study

Status: **COMPLETED_NON_SUCCESS**.

The original protocol and final prospective clarifications are preserved as separate files. New P1 is distinct from rejected v4 P1.

## Phase decisions

{
  "calibration": "PASSED",
  "common_acquisition": "PASSED",
  "paired_acquisition": "COMPLETED",
  "confirmation": "COMPLETED",
  "reference": "COMPLETED",
  "ablation": "COMPLETED",
  "profiling": "COMPLETED",
  "analysis": "COMPLETED"
}

## Development/native gate

```json
{
  "passes": true,
  "strata": {
    "receiver_anchored_v1/P0/L16": {
      "n": 64,
      "current_accuracy": 1.0,
      "robust_accuracy": 1.0,
      "mean_CE": 0.00026367387962476216,
      "horizon_CE": {
        "0": 0.00026580574171930493,
        "1": 0.00026348088613303844,
        "4": 0.0002619045170604295,
        "8": 0.0002647144905267851
      },
      "native_prefix_mean_CE": 0.0002723783638458599,
      "passes": true
    },
    "receiver_anchored_v1/P0/L24": {
      "n": 64,
      "current_accuracy": 1.0,
      "robust_accuracy": 1.0,
      "mean_CE": 0.0002628292442402787,
      "horizon_CE": {
        "0": 0.0002635451689911861,
        "1": 0.00026310369685234036,
        "4": 0.0002626511285370725,
        "8": 0.0002637879243820862
      },
      "native_prefix_mean_CE": 0.0002665067398197607,
      "passes": true
    },
    "receiver_anchored_v1/P1/L16": {
      "n": 64,
      "current_accuracy": 1.0,
      "robust_accuracy": 1.0,
      "mean_CE": 0.0002702548766752871,
      "horizon_CE": {
        "0": 0.0002716281371704099,
        "1": 0.00026969805435328453,
        "4": 0.0002683618453147574,
        "8": 0.00027198456859878206
      },
      "native_prefix_mean_CE": 0.0002782044979880993,
      "passes": true
    },
    "receiver_anchored_v1/P1/L24": {
      "n": 64,
      "current_accuracy": 1.0,
      "robust_accuracy": 1.0,
      "mean_CE": 0.00027027186695918585,
      "horizon_CE": {
        "0": 0.00026962430274579674,
        "1": 0.00027040078884965624,
        "4": 0.0002697573434033984,
        "8": 0.0002714219626795966
      },
      "native_prefix_mean_CE": 0.00027272398601970355,
      "passes": true
    }
  }
}
```

## Selection and final acceptance

```json
{
  "update": 512,
  "eligible": [
    256,
    512
  ],
  "native_only": true,
  "summaries": {
    "256": {
      "T": {
        "passes": true,
        "strata": {
          "receiver_anchored_v1/P0/L16": {
            "n": 128,
            "current_accuracy": 1.0,
            "robust_accuracy": 1.0,
            "mean_CE": 0.00018864945825297955,
            "horizon_CE": {
              "0": 0.00019037833209267774,
              "1": 0.00018870790322012,
              "4": 0.00018860916983953757,
              "8": 0.00018792896432273665
            },
            "native_prefix_mean_CE": 0.00019398164788597683,
            "passes": true
          },
          "receiver_anchored_v1/P0/L24": {
            "n": 128,
            "current_accuracy": 1.0,
            "robust_accuracy": 1.0,
            "mean_CE": 0.0001886210121610662,
            "horizon_CE": {
              "0": 0.00018843791667677579,
              "1": 0.00018852493025178774,
              "4": 0.00018877164580999306,
              "8": 0.00018758209699853978
            },
            "native_prefix_mean_CE": 0.0001904014414340812,
            "passes": true
          },
          "receiver_anchored_v1/P1/L16": {
            "n": 128,
            "current_accuracy": 1.0,
            "robust_accuracy": 1.0,
            "mean_CE": 0.00019422383177042694,
            "horizon_CE": {
              "0": 0.00019530186142446837,
              "1": 0.00019415892970187087,
              "4": 0.00019404295414915396,
              "8": 0.0001938273436792315
            },
            "native_prefix_mean_CE": 0.00019867190540523438,
            "passes": true
          },
          "receiver_anchored_v1/P1/L24": {
            "n": 128,
            "current_accuracy": 1.0,
            "robust_accuracy": 1.0,
            "mean_CE": 0.00019464013771606486,
            "horizon_CE": {
              "0": 0.00019482788832192455,
              "1": 0.0001944074746518254,
              "4": 0.00019441253334662179,
              "8": 0.00019376117597857956
            },
            "native_prefix_mean_CE": 0.0001954598071544448,
            "passes": true
          }
        }
      },
      "C": {
        "passes": true,
        "strata": {
          "receiver_anchored_v1/P0/L16": {
            "n": 128,
            "current_accuracy": 1.0,
            "robust_accuracy": 1.0,
            "mean_CE": 0.0001886265566714856,
            "horizon_CE": {
              "0": 0.00019033363122389346,
              "1": 0.00018867484072870866,
              "4": 0.0001885872813147671,
              "8": 0.00018791871585221998
            },
            "native_prefix_mean_CE": 0.00019399922054219587,
            "passes": true
          },
          "receiver_anchored_v1/P0/L24": {
            "n": 128,
            "current_accuracy": 1.0,
            "robust_accuracy": 1.0,
            "mean_CE": 0.00018859175657539977,
            "horizon_CE": {
              "0": 0.0001884090444264075,
              "1": 0.00018849466144388316,
              "4": 0.0001887199598229472,
              "8": 0.00018755788079261038
            },
            "native_prefix_mean_CE": 0.0001903987597984269,
            "passes": true
          },
          "receiver_anchored_v1/P1/L16": {
            "n": 128,
            "current_accuracy": 1.0,
            "robust_accuracy": 1.0,
            "mean_CE": 0.00019422832093596298,
            "horizon_CE": {
              "0": 0.0001953111709553923,
              "1": 0.00019416032415620066,
              "4": 0.00019404015853297096,
              "8": 0.00019383339318324033
            },
            "native_prefix_mean_CE": 0.00019866817934399705,
            "passes": true
          },
          "receiver_anchored_v1/P1/L24": {
            "n": 128,
            "current_accuracy": 1.0,
            "robust_accuracy": 1.0,
            "mean_CE": 0.00019464780304441078,
            "horizon_CE": {
              "0": 0.00019483905902006882,
              "1": 0.00019441352395688227,
              "4": 0.00019441904805717058,
              "8": 0.0001937667598781445
            },
            "native_prefix_mean_CE": 0.0001954590294393673,
            "passes": true
          }
        }
      }
    },
    "512": {
      "T": {
        "passes": true,
        "strata": {
          "receiver_anchored_v1/P0/L16": {
            "n": 128,
            "current_accuracy": 1.0,
            "robust_accuracy": 1.0,
            "mean_CE": 0.00013986901646322782,
            "horizon_CE": {
              "0": 0.0001409289615708076,
              "1": 0.0001399121276506321,
              "4": 0.00013991304274441063,
              "8": 0.00013944092569317945
            },
            "native_prefix_mean_CE": 0.00014346732313796906,
            "passes": true
          },
          "receiver_anchored_v1/P0/L24": {
            "n": 128,
            "current_accuracy": 1.0,
            "robust_accuracy": 1.0,
            "mean_CE": 0.000139852572076774,
            "horizon_CE": {
              "0": 0.00013977805832610102,
              "1": 0.00013982459182670937,
              "4": 0.00013987205792886925,
              "8": 0.00013917412800878992
            },
            "native_prefix_mean_CE": 0.0001410430483990884,
            "passes": true
          },
          "receiver_anchored_v1/P1/L16": {
            "n": 128,
            "current_accuracy": 1.0,
            "robust_accuracy": 1.0,
            "mean_CE": 0.00014421391833085341,
            "horizon_CE": {
              "0": 0.0001448667310910423,
              "1": 0.00014416370834169356,
              "4": 0.00014412503779226427,
              "8": 0.000144033284584566
            },
            "native_prefix_mean_CE": 0.00014729931243806504,
            "passes": true
          },
          "receiver_anchored_v1/P1/L24": {
            "n": 128,
            "current_accuracy": 1.0,
            "robust_accuracy": 1.0,
            "mean_CE": 0.00014452324731952532,
            "horizon_CE": {
              "0": 0.00014468885774476803,
              "1": 0.0001443689929772063,
              "4": 0.00014437687408985767,
              "8": 0.00014395875669492852
            },
            "native_prefix_mean_CE": 0.0001450790304341846,
            "passes": true
          }
        }
      },
      "C": {
        "passes": true,
        "strata": {
          "receiver_anchored_v1/P0/L16": {
            "n": 128,
            "current_accuracy": 1.0,
            "robust_accuracy": 1.0,
            "mean_CE": 0.0001398730651825484,
            "horizon_CE": {
              "0": 0.00014093733773279382,
              "1": 0.00013991212318842372,
              "4": 0.00013991490021680875,
              "8": 0.0001394451114720141
            },
            "native_prefix_mean_CE": 0.00014348169556299695,
            "passes": true
          },
          "receiver_anchored_v1/P0/L24": {
            "n": 128,
            "current_accuracy": 1.0,
            "robust_accuracy": 1.0,
            "mean_CE": 0.00013985749697386542,
            "horizon_CE": {
              "0": 0.00013977432860201588,
              "1": 0.00013982738025219987,
              "4": 0.0001398767086868702,
              "8": 0.00013917505447125222
            },
            "native_prefix_mean_CE": 0.00014105693507104888,
            "passes": true
          },
          "receiver_anchored_v1/P1/L16": {
            "n": 128,
            "current_accuracy": 1.0,
            "robust_accuracy": 1.0,
            "mean_CE": 0.00014421851754859208,
            "horizon_CE": {
              "0": 0.0001448676614472788,
              "1": 0.00014416649963777672,
              "4": 0.00014412037990041426,
              "8": 0.0001440398002330312
            },
            "native_prefix_mean_CE": 0.00014728883579095964,
            "passes": true
          },
          "receiver_anchored_v1/P1/L24": {
            "n": 128,
            "current_accuracy": 1.0,
            "robust_accuracy": 1.0,
            "mean_CE": 0.00014453211854756505,
            "horizon_CE": {
              "0": 0.00014469909831404948,
              "1": 0.00014437969932146189,
              "4": 0.00014438385568382728,
              "8": 0.00014396480685263668
            },
            "native_prefix_mean_CE": 0.00014507092014544773,
            "passes": true
          }
        }
      }
    }
  },
  "acquisition_successful": true
}
```
```json
{
  "primary_delta": 0.0,
  "primary_interval": [
    0.0,
    0.0
  ],
  "primary_pass": false,
  "length_specific": {
    "16": {
      "point": 0.0,
      "interval": [
        0.0,
        0.0
      ]
    },
    "24": {
      "point": 0.0,
      "interval": [
        0.0,
        0.0
      ]
    }
  },
  "native": {
    "P0/L16": {
      "current_accuracy": [
        1.0,
        1.0
      ],
      "robust_accuracy": [
        1.0,
        1.0
      ],
      "horizon_CE": [
        [
          0.00014131453045251874,
          0.00014023084547432063,
          0.0001398045803853165,
          0.0001400024315501014
        ],
        [
          0.00014130800738598737,
          0.00014023503153737238,
          0.00013980760208198717,
          0.00014000824712923077
        ]
      ],
      "native_mean_CE": [
        0.00013998963552808164,
        0.00013999228760460083
      ],
      "native_mean_CE_difference": -2.6520765191958336e-09,
      "paired_native_accuracy_difference": 0.0,
      "adjusted_interval": [
        -0.01697162304112898,
        0.01697162304112898
      ],
      "zero_discordance_bound": 0.01697162304112898,
      "paired_CE_differences": [
        6.523066531372024e-09,
        -4.186063051747624e-09,
        -3.0216966706575477e-09,
        -5.815579129375692e-09
      ],
      "switched_T_accuracy": 1.0,
      "switched_T_robust": 1.0,
      "cross_program_CE_deterioration": [
        2.421359113213839e-08,
        -1.3736368487116124e-08,
        -2.6541073339103605e-08,
        -1.6763053167778708e-08
      ]
    },
    "P0/L24": {
      "current_accuracy": [
        1.0,
        1.0
      ],
      "robust_accuracy": [
        1.0,
        1.0
      ],
      "horizon_CE": [
        [
          0.00014006902628693751,
          0.00013975497066098796,
          0.00013956336663056845,
          0.0001395926846043949
        ],
        [
          0.000140080661481079,
          0.0001397645110614576,
          0.00013956755201149917,
          0.0001395996637540975
        ]
      ],
      "native_mean_CE": [
        0.00013963955869906598,
        0.00013964659269430845
      ],
      "native_mean_CE_difference": -7.0339952424681215e-09,
      "paired_native_accuracy_difference": 0.0,
      "adjusted_interval": [
        -0.01697162304112898,
        0.01697162304112898
      ],
      "zero_discordance_bound": 0.01697162304112898,
      "paired_CE_differences": [
        -1.1635194141490501e-08,
        -9.540400469632004e-09,
        -4.185380930721294e-09,
        -6.979149702601717e-09
      ],
      "switched_T_accuracy": 1.0,
      "switched_T_robust": 1.0,
      "cross_program_CE_deterioration": [
        4.19143475482997e-09,
        -1.327175880305731e-08,
        -2.3981598928912717e-08,
        -2.2118555875749735e-08
      ]
    },
    "P1/L16": {
      "current_accuracy": [
        1.0,
        1.0
      ],
      "robust_accuracy": [
        1.0,
        1.0
      ],
      "horizon_CE": [
        [
          0.00014558519035290374,
          0.00014432179125378752,
          0.00014414157146802609,
          0.00014451867056664014
        ],
        [
          0.00014559496599986232,
          0.0001443266782388264,
          0.00014414878610580217,
          0.00014452518665564185
        ]
      ],
      "native_mean_CE": [
        0.00014428646887702546,
        0.0001442921361408079
      ],
      "native_mean_CE_difference": -5.667263782446839e-09,
      "paired_native_accuracy_difference": 0.0,
      "adjusted_interval": [
        -0.01697162304112898,
        0.01697162304112898
      ],
      "zero_discordance_bound": 0.01697162304112898,
      "paired_CE_differences": [
        -9.775646958587458e-09,
        -4.886985038865532e-09,
        -7.214637776087329e-09,
        -6.5160890017068596e-09
      ],
      "switched_T_accuracy": 1.0,
      "switched_T_robust": 1.0,
      "cross_program_CE_deterioration": [
        -1.397353344145813e-08,
        -4.4223469330972875e-09,
        1.2571319984999718e-08,
        -1.1626042351053911e-09
      ]
    },
    "P1/L24": {
      "current_accuracy": [
        1.0,
        1.0
      ],
      "robust_accuracy": [
        1.0,
        1.0
      ],
      "horizon_CE": [
        [
          0.00014465719164036273,
          0.0001443221863155486,
          0.00014407959831430617,
          0.00014443318109158554
        ],
        [
          0.00014466044819982926,
          0.0001443280042110473,
          0.0001440903045875075,
          0.00014444551705139475
        ]
      ],
      "native_mean_CE": [
        0.00014430714110142832,
        0.00014431587550401513
      ],
      "native_mean_CE_difference": -8.734402586802912e-09,
      "paired_native_accuracy_difference": 0.0,
      "adjusted_interval": [
        -0.01697162304112898,
        0.01697162304112898
      ],
      "zero_discordance_bound": 0.01697162304112898,
      "paired_CE_differences": [
        -3.256559466535691e-09,
        -5.81789549869427e-09,
        -1.0706273201321892e-08,
        -1.2335959809206543e-08
      ],
      "switched_T_accuracy": 1.0,
      "switched_T_robust": 1.0,
      "cross_program_CE_deterioration": [
        -1.8626053588377545e-08,
        -6.512905770250654e-09,
        -5.117442469781963e-09,
        -6.746887493136455e-09
      ]
    }
  },
  "native_competence": true,
  "native_comparability": true,
  "treatment_compatibility": true,
  "alternating_robust": {
    "L16/00": 1.0,
    "L16/01": 1.0,
    "L16/10": 1.0,
    "L16/11": 1.0,
    "L24/00": 1.0,
    "L24/01": 1.0,
    "L24/10": 1.0,
    "L24/11": 1.0
  },
  "robust_patterns": {
    "T": {
      "0000": 0,
      "0001": 0,
      "0010": 0,
      "0011": 0,
      "0100": 0,
      "0101": 0,
      "0110": 0,
      "0111": 0,
      "1000": 0,
      "1001": 0,
      "1010": 0,
      "1011": 0,
      "1100": 0,
      "1101": 0,
      "1110": 0,
      "1111": 512
    },
    "C": {
      "0000": 0,
      "0001": 0,
      "0010": 0,
      "0011": 0,
      "0100": 0,
      "0101": 0,
      "0110": 0,
      "0111": 0,
      "1000": 0,
      "1001": 0,
      "1010": 0,
      "1011": 0,
      "1100": 0,
      "1101": 0,
      "1110": 0,
      "1111": 512
    }
  },
  "current_patterns": {
    "T": {
      "0000": 0,
      "0001": 0,
      "0010": 0,
      "0011": 0,
      "0100": 0,
      "0101": 0,
      "0110": 0,
      "0111": 0,
      "1000": 0,
      "1001": 0,
      "1010": 0,
      "1011": 0,
      "1100": 0,
      "1101": 0,
      "1110": 0,
      "1111": 512
    },
    "C": {
      "0000": 0,
      "0001": 0,
      "0010": 0,
      "0011": 0,
      "0100": 0,
      "0101": 0,
      "0110": 0,
      "0111": 0,
      "1000": 0,
      "1001": 0,
      "1010": 0,
      "1011": 0,
      "1100": 0,
      "1101": 0,
      "1110": 0,
      "1111": 512
    }
  },
  "continuation_metrics": {
    "T/L16/00/boundary": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00014023084547432063
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.0001398045803853165
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.0001400024315501014
      }
    },
    "T/L16/00/alternating": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00014429920658187712
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.000139792476957723
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.0001400226879013644
      }
    },
    "T/L16/01/boundary": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00014431736890685443
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00014415414278801109
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.00014451750796240503
      }
    },
    "T/L16/01/alternating": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00014024784211130736
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00014415716576365867
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.00014447955831542458
      }
    },
    "T/L16/10/boundary": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00014021710910583352
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.0001397780393119774
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.0001399856684969336
      }
    },
    "T/L16/10/alternating": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00014430269746412705
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.0001397708250578944
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.00014000801965607934
      }
    },
    "T/L16/11/boundary": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00014432179125378752
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00014414157146802609
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.00014451867056664014
      }
    },
    "T/L16/11/alternating": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.0001402427195813516
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00014414412859764525
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.00014447909131831693
      }
    },
    "T/L24/00/boundary": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00013975497066098796
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00013956336663056845
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.0001395926846043949
      }
    },
    "T/L24/00/alternating": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.0001442965795206419
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00013958222594112613
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.00013960083434483295
      }
    },
    "T/L24/01/boundary": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00014431567340977836
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00014407448087183639
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.0001444264342040924
      }
    },
    "T/L24/01/alternating": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.0001397661475550649
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00014405701737985055
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.00014439290866619103
      }
    },
    "T/L24/10/boundary": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.0001397416989021849
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00013953938503163954
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.00013957056604851914
      }
    },
    "T/L24/10/alternating": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00014429936777560215
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00013956196991671277
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.00013958407026848363
      }
    },
    "T/L24/11/boundary": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.0001443221863155486
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00014407959831430617
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.00014443318109158554
      }
    },
    "T/L24/11/alternating": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00013975241050445675
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.0001440600393607383
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.0001444043118681293
      }
    },
    "C/L16/00/boundary": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00014023503153737238
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00013980760208198717
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.00014000824712923077
      }
    },
    "C/L16/00/alternating": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00014430269500564918
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00013979619730264403
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.00014003292633901765
      }
    },
    "C/L16/01/boundary": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00014432295277799767
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00014416368397007773
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.0001445256524590377
      }
    },
    "C/L16/01/alternating": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00014025272655260324
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.0001441653095497486
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.0001444923583022728
      }
    },
    "C/L16/10/boundary": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00014023386702888274
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00013978967587036095
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.00013999451098811733
      }
    },
    "C/L16/10/alternating": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00014431084342447775
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00013978432342298674
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.0001400233809221163
      }
    },
    "C/L16/11/boundary": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.0001443266782388264
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00014414878610580217
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.00014452518665564185
      }
    },
    "C/L16/11/alternating": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00014026320279469928
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00014415367094500198
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.0001444900302658425
      }
    },
    "C/L24/00/boundary": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.0001397645110614576
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00013956755201149917
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.0001395996637540975
      }
    },
    "C/L24/00/alternating": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00014431031013373286
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00013958990402329619
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.000139612237077813
      }
    },
    "C/L24/01/boundary": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00014432358386784472
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00014408262471476974
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.0001444387681885928
      }
    },
    "C/L24/01/alternating": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00013977964569278356
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.0001440656267277518
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.00014440594097209214
      }
    },
    "C/L24/10/boundary": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.00013975822432144014
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.0001395496244356309
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.0001395870911409247
      }
    },
    "C/L24/10/alternating": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.0001443054183738468
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.000139574537300291
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.00013959943133556862
      }
    },
    "C/L24/11/boundary": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.0001443280042110473
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.0001440903045875075
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.00014444551705139475
      }
    },
    "C/L24/11/alternating": {
      "1": {
        "accuracy": 1.0,
        "CE": 0.0001397728931920028
      },
      "4": {
        "accuracy": 1.0,
        "CE": 0.00014406934855060172
      },
      "8": {
        "accuracy": 1.0,
        "CE": 0.00014441594883862763
      }
    }
  },
  "ablation": {
    "point": [
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "lower": [
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "passes": [
      false,
      false,
      false,
      false
    ],
    "family_size": 4,
    "quantile": 0.0125,
    "bank": "f8f083f2f00f4c3d935f7801aee0b6c27887ad3598bd54ade6511b9902285e64"
  },
  "latency": {
    "order": [
      "P0/L16",
      "P0/L24",
      "P1/L16",
      "P1/L24"
    ],
    "point": [
      1.0141067239090709,
      1.001952763432633,
      1.0051003375244056,
      0.9994180102292951
    ],
    "upper": [
      1.0442773396808784,
      1.0080557873126255,
      1.0118700807453325,
      1.0045221880496116
    ],
    "passes": [
      true,
      true,
      true,
      true
    ],
    "family_size": 4,
    "quantile": 0.9875,
    "bank": "05da726cdecdeb413c94d95248e49e49708dfaf4c99feab338aefbdbe2e37a43"
  },
  "native_only_checkpoint_eligible": true,
  "conditional_input_sampling_uncertainty": true,
  "analysis_seconds": 0.5489300601184368,
  "bootstrap_replicates": 20000,
  "bootstrap_seed": 6010,
  "separate_ablation_latency_families": true,
  "confirmation_bank": "f8f083f2f00f4c3d935f7801aee0b6c27887ad3598bd54ade6511b9902285e64",
  "profiling_bank": "05da726cdecdeb413c94d95248e49e49708dfaf4c99feab338aefbdbe2e37a43",
  "full_success": false
}
```

## Actual resources

```json
{
  "study_attempted_tokens": 533776,
  "study_completed_tokens": 533776,
  "study_failed_tokens": 0,
  "study_selected_calls": 4270208,
  "overall_completed_tokens": 534162,
  "overall_attempted_tokens": 534163,
  "overall_selected_calls": 4273296,
  "study_commit_attempts": 533776,
  "study_commits": 533776,
  "backward_evaluations": 3072,
  "optimizer_updates": 1536,
  "phase_completed_tokens": {
    "calibration": 2064,
    "common_training": 40960,
    "development": 9216,
    "T_training": 57344,
    "C_training": 57344,
    "selection": 73728,
    "confirmation": 174080,
    "reference": 18432,
    "ablation": 73728,
    "profiling": 26880
  },
  "phase_attempted_tokens": {
    "calibration": 2064,
    "common_training": 40960,
    "development": 9216,
    "T_training": 57344,
    "C_training": 57344,
    "selection": 73728,
    "confirmation": 174080,
    "reference": 18432,
    "ablation": 73728,
    "profiling": 26880
  },
  "contextual_commit_calls": {
    "SWA/bank": 1067552,
    "GQA/bank": 1067552,
    "GDN/bank": 1067552,
    "SWA/prelude": 533776
  },
  "selected_block_calls": {
    "0": 533776,
    "1": 533776,
    "2": 533776,
    "3": 787728,
    "4": 279824,
    "5": 533776,
    "6": 533776,
    "7": 533776
  },
  "profiling_warmup_tokens": 4480,
  "elapsed_accelerator_window_seconds": 7573.8967925217,
  "carried_prerequisite_seconds": 22.66048240661621,
  "carried_prerequisite_ledger": {
    "attempted_tokens": 387,
    "completed_tokens": 386,
    "failed_tokens": 1,
    "selected_bank_calls": 3088,
    "completed_bank_calls": 3088,
    "attempted_terminal_commits": 386,
    "terminal_commits": 386,
    "backward_evaluations": 13,
    "optimizer_updates": 0,
    "warmup_tokens": 0,
    "token_ceiling": 512,
    "block_ceiling": 4096
  },
  "peak_memory_allocated": 311910912,
  "peak_memory_reserved": 316669952,
  "full_confirmation_state_files": 1024
}
```

## Integrity

```json
{
  "protected_count": 776,
  "changed_or_missing": [],
  "unexpected_outside": [],
  "tracked_diff_sha256": "45b3d093bd7fd7976019bf4572dc574784ed271b3294b86194f80c05826a8eac",
  "tracked_diff_unchanged": true,
  "historical_exception": "experiments/results/gatec_fresh_task_discovery_4700_v1/artifact_integrity_incident.json retained unchanged; no cache restoration or historical rewriting",
  "success": true
}
```

Task errors do not remove cases. Unexecuted phases have no scientific verdict. No results from prior attempts or prerequisites are pooled into outcomes.
Unknown Stage1 exposure remains a limitation; held-out means disjoint from documented retained acquisition/selection inputs.
Unsupported FLOPs are not estimated. Statistical intervals are conditional on these models, not training-seed replication.
No production edits, prior artifact overwrite, selector, additional variant, replacement seed or Git commit is permitted.
Stop boundary: this single study. No automatic retry or follow-up is authorized.
