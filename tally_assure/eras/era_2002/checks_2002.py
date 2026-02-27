from __future__ import annotations 

from pathlib import Path 
from typing import Any ,Dict ,Optional 

import pandas as pd 

# Import shared atomic parsers / helpers from the core checksums module.
# This module is imported lazily from tally_assure.checksums to avoid import cycles.
from ...checksums import _read_atomic_candidate_totals ,_read_atomic_party_totals 


def _fail_provided_vs_qa (key :str ,provided :float ,qa :float )->Dict [str ,Any ]:
    provided =float (provided )
    qa =float (qa )
    pct =(provided /qa *100 )if qa !=0 else (0 if provided ==0 else float ("inf"))
    return {
    "key":key ,
    "provided_value":provided ,
    "qa_value":qa ,
    "diff_provided_minus_qa":(provided -qa ),
    "provided_as_pct_of_qa":pct ,
    }


def checksum_splitvote_endstate_2002 (
split_endstate_csv :Path ,
candidate_csv :Path ,
party_csv :Path ,
)->Dict [str ,Any ]:
    """Checks the 2002-era approval-ballot solution matrix against atomic candidate/party totals."""
    mat =pd .read_csv (split_endstate_csv )

    required ={
    "Party",
    "Sum_from_split_vote_counts",
    "Total Party Votes",
    "QA_Total_Party_Votes_from_atomic_party",
    "consistent",
    }
    if not required .issubset (set (mat .columns )):
        return {
        "file":str (split_endstate_csv ),
        "error":f"Missing required columns: {sorted(required - set(mat.columns))}",
        "checks":{},
        }

    def is_summary_party (p :str )->bool :
        p =str (p ).strip ().lower ()
        return p in {
        "sum_from_split_vote_counts",
        "party vote totals",# legacy name
        "provided candidate split vote totals",
        "qa sums from the candidate csv",
        "consistent",
        }

    main =mat [~mat ["Party"].astype (str ).apply (is_summary_party )].copy ()

    bookkeeping =[
    "Sum_from_split_vote_counts",
    "Total Party Votes",
    "QA_Total_Party_Votes_from_atomic_party",
    "consistent",
    ]
    count_cols =[c for c in mat .columns if c not in ["Party"]+bookkeeping ]

    # numeric conversion for main rows
    for c in count_cols +["Sum_from_split_vote_counts","Total Party Votes"]:
        main .loc [:,c ]=pd .to_numeric (main [c ],errors ="coerce").fillna (0 )

        # Row check: sum of row counts == provided Total Party Votes
    row_fail =[]
    row_pass =[]
    for _ ,r in main .iterrows ():
        party =str (r ["Party"]).strip ()
        qa_sum =float (r [count_cols ].sum ())
        provided_total =float (r ["Total Party Votes"])
        if qa_sum ==provided_total :
            row_pass .append ({"party":party })
        else :
            row_fail .append ({"party":party ,**_fail_provided_vs_qa ("Total Party Votes",provided_total ,qa_sum )})

            # Column check: sum of column counts across parties == atomic candidate totals (from candidate CSV totals row)
    atomic_cands =_read_atomic_candidate_totals (candidate_csv )
    col_fail =[]
    col_pass =[]
    for c in count_cols :
        base =str (c ).strip ()
        base2 =base .rsplit ("(",1 )[0 ].strip ()if "("in base and base .endswith (")")else base 
        lookup_keys =[base ,base2 ," ".join (base2 .split ())]
        official_key =next ((k for k in lookup_keys if k in atomic_cands ),None )
        if official_key is None :
            continue 
        provided =float (main [c ].sum ())
        qa =float (atomic_cands [official_key ])
        if provided ==qa :
            col_pass .append ({"key":str (c )})
        else :
            col_fail .append (_fail_provided_vs_qa (str (c ),provided ,qa ))

            # Party totals per row vs atomic party totals (from party CSV totals row)
    atomic_party =_read_atomic_party_totals (party_csv )
    party_fail =[]
    party_pass =[]
    for _ ,r in main .iterrows ():
        party =str (r ["Party"]).strip ()
        party_key =party .split (" (",1 )[0 ].strip ()if " ("in party else party 
        if party_key not in atomic_party :
            continue 
        provided =float (r ["Total Party Votes"])
        qa =float (atomic_party [party_key ])
        if provided ==qa :
            party_pass .append ({"party":party })
        else :
            party_fail .append (_fail_provided_vs_qa (party ,provided ,qa ))

            # Provided totals row vs sum row (elementwise)
    def find_row (label :str )->Optional [pd .Series ]:
        hit =mat [mat ["Party"].astype (str ).str .strip ().str .lower ()==label .lower ()]
        return hit .iloc [0 ]if len (hit )else None 

    sum_row =find_row ("Sum_from_split_vote_counts")
    prov_row = find_row("Provided candidate split vote totals")
    if prov_row is None:
        prov_row = find_row("Party vote totals")
    totals_row_fail =[]
    totals_row_pass =[]
    if sum_row is not None and prov_row is not None :
        for c in count_cols +["Total Party Votes","Sum_from_split_vote_counts","QA_Total_Party_Votes_from_atomic_party"]:
            qa =float (pd .to_numeric (sum_row .get (c ,0 ),errors ="coerce")or 0 )
            provided =float (pd .to_numeric (prov_row .get (c ,0 ),errors ="coerce")or 0 )
            if provided ==qa :
                totals_row_pass .append ({"key":c })
            else :
                totals_row_fail .append (_fail_provided_vs_qa (c ,provided ,qa ))
    else :
        totals_row_fail .append ({"key":"provided_totals_row_missing_or_sum_row_missing"})

        # Corner cell check (Consistent row × consistent column) must be string 'true'/'false'/'error'
    consistent_row =find_row ("Consistent")
    consistent_checks_pass =[]
    consistent_checks_fail =[]
    if consistent_row is None :
        consistent_checks_fail .append ({"key":"corner","value":"error","reason":"Consistent row missing"})
    else :
        val =consistent_row .get ("consistent")
        sval =str (val ).strip ().casefold ()if val is not None else "error"
        if sval =="true":
            consistent_checks_pass .append ({"key":"corner","value":"true"})
        elif sval =="false":
            consistent_checks_fail .append ({"key":"corner","value":"false"})
        else :
            consistent_checks_fail .append ({"key":"corner","value":"error","reason":"unable to compute corner cell"})

    return {
    "file":str (split_endstate_csv ),
    "checks":{
    "splitvote_row_consistency_bool":{"passed":row_pass ,"failed":row_fail },
    "splitvote_candidate_column_sums_vs_atomic_candidate_totals":{"passed":col_pass ,"failed":col_fail },
    "splitvote_party_totals_vs_atomic_party_totals":{"passed":party_pass ,"failed":party_fail },
    "splitvote_provided_totals_row_vs_sum_of_rows":{"passed":totals_row_pass ,"failed":totals_row_fail },
    "splitvote_endstate_consistent_row":{"passed":consistent_checks_pass ,"failed":consistent_checks_fail },
    },
    }
