import os
import json
import re
import csv
import argparse
from collections import defaultdict

# --- Step 1: Preprocessing and CSV generation ---
legal_name_list = [
    't1_c95k50u', 't1_c95k75v', 't1_c95mz3n', 't1_c95tixb', 't1_c95l4my',
    't1_c95mcms', 't1_c95mdhe', 't3_1aenyc', 't1_c8wpqjn', 't1_c8wrovj',
    't1_c8wuxh2', 't1_cssxisl', 't1_c9zukmv', 't1_cjy73pb', 't1_cjy865a',
    't1_cjydv04', 't1_cjy9w7i', 't1_cg9ansq', 't1_cg7ybnn', 't1_cm5lbyg',
    't1_cm5lvxq', 't1_cm5mup4', 't1_cm5xfbn', 't1_cm64nte', 't1_cm5mwyw',
    't1_cm5nasq', 't1_cm5niv6', 't1_cm5nwf0', 't1_cm5nyuj', 't3_1pg4zf',
    't1_cd2oaft', 't1_cd2rjoh', 't1_cd2ssov', 't3_2x4qkh', 't1_cowyuji',
    't1_cg5uill', 't1_cg5u5za', 't1_cg5vvtk', 't1_cg676fl', 't1_cg6891i',
    't1_cs8qzp2', 't1_cs8wghl', 't1_cs8ywso', 't1_cs96nzv', 't3_2cy7r8',
    't1_cjkhajj', 't1_cjkh0e7', 't1_cr706at', 't1_cr71uyn', 't1_c9vagau',
    't3_34yo0h', 't1_cqze8zq', 't1_cqzdpbc', 't1_chsthjv', 't1_chstlop',
    't1_chstdo9', 't1_chstnie', 't1_chtmsji', 't1_cf6guot', 't1_cf6jm5k',
    't1_cf6qwoo', 't1_clik16y', 't1_clikadz', 't1_clipfjz', 't1_cliyfh1',
    't1_cstu94m', 't1_csuck5d', 't1_cstvbko', 't1_cstvlvt', 't1_cstwvof',
    't1_csudcox', 't1_csvrg7x', 't1_cstvopg', 't1_cstx7qd', 't1_csuddim',
    't1_cstvpcf', 't1_cstvpqw', 't1_csu57ld', 't1_csu7lg5', 't1_csu8gdo',
    't1_csudgkr', 't1_csu0f6s', 't1_csv5ef5', 't1_csw1xe2', 't1_csu1ujy',
    't1_csu57h5', 't1_csudikx', 't3_1bcfz2', 't1_c95qp1r', 't1_ctl8trd',
    't1_ctlr9v3', 't1_ctlv7xc', 't1_ctltslz', 't1_cb3jjs2', 't1_cb3kir7',
    't1_cb3nynv', 't1_cb3qcwp', 't1_cb3tns6', 't1_cb3zy7m', 't1_cb4mafq',
    't1_cb4n4oh', 't1_cb46pb4', 't1_cb3qdsx', 't1_cb3un1d', 't1_cb3q90x',
    't1_cb3vnb9', 't1_cbcwihs', 't1_cb4019l', 't1_cb49a96', 't1_cbcwm6p',
    't3_1r3id3', 't3_1fqssq', 't1_cacvan5', 't1_cacvmf2', 't1_caf1ab9',
    't3_1qm5d4', 't1_cde6meq', 't1_cde6orn', 't1_cdeaz71', 't1_cdekndg',
    't3_1c4do4', 't1_c9cwx8n', 't1_c9dla74', 't1_c9d39gw', 't1_c9d5s0t',
    't1_cc5kmq0', 't1_cml0lai', 't1_cml2ft5', 't1_cml53oh', 't1_cml60ij',
    't1_cthvpeo', 't1_ctiqpyy', 't1_ctita17', 't1_ctisowf', 't1_ckxzlnz',
    't1_ckxzyeu', 't1_cky1afp', 't1_cky0d05', 't1_cky0fhe', 't1_cky0hj9',
    't1_cky2oax', 't1_cky2xto', 't1_ckybda8', 't1_ckzm1sy', 't1_ckyceyn',
    't1_cky4g8p', 't1_cky7neb', 't1_cl0z8pc', 't1_cky6lbd', 't1_cky9oqt',
    't1_cky3f7y', 't1_cky44de', 't1_ckylfmb', 't1_ckyblqs', 't1_ckydk5a',
    't1_ccpsrei', 't1_ccpv6xh', 't1_ccq583t', 't1_ccqbyt1', 't1_cc12s78',
    't3_2nck53', 't1_cmcfad1', 't1_cmcuo78', 't1_cmdaz4s', 't1_cmcyo3d',
    't1_cmcfx2p', 't1_cmckzsq', 't1_cmcetow', 't1_cmcf7ri', 't1_cmcfjhb',
    't1_cmcgspm', 't1_cmcn89s', 't1_cmddi26', 't1_cmde47f', 't1_cmeg0mb',
    't1_cmcmrsu', 't1_cmcnb1u', 't1_cmculcq', 't1_cmcvtgd', 't1_cmcz2h1',
    't1_cmdpt23', 't1_cmehocs', 't1_cml5ds9', 't1_cmcg1sw', 't1_cmdn3uc',
    't1_cmclswv', 't1_cmcnu78', 't1_cmcuc2d', 't1_cmdxdee', 't1_cmedpvb',
    't1_cmcpg05', 't1_cmcue6k', 't1_cmcztsw', 't1_cmdfadm', 't1_cmdtn63',
    't1_cmeg54f', 't1_cmenx3z', 't1_cmdamo2', 't1_cmdupvw', 't1_cml4gcq',
    't1_cmlh163', 't1_cmfx2yt', 't1_cpm4kuv', 't1_cpm4zcb', 't1_cpluhpn',
    't1_cplwg2z', 't1_cpm1b9f', 't1_cpmoklh', 't1_cpmtijj', 't1_cai8u8v',
    't1_cai8wyf', 't1_caie40w', 't1_caiixhl', 't1_c8vtkju', 't1_c8vrprg',
    't1_c8vsl8l', 't1_c8vuv93', 't1_c8wq24a', 't3_1h8unp', 't1_cary5i8',
    't1_carykfz', 't1_caryrhj', 't1_cas5hhz', 't1_casyz2m', 't1_caryrmq',
    't1_cti7mh5', 't1_ctiqgmx', 't1_ctirspu', 't1_ctitysx', 't1_ctj3ddp',
    't1_ctidpo5', 't1_ctiqjtn', 't1_ctilqsl', 't1_ctiq2mi', 't3_1grwqj',
    't1_can6ouc', 't1_can6svh', 't1_canbhof', 't1_canhzf8', 't1_cani2rw',
    't1_cao0soi', 't1_can6q8n', 't3_1nh3z4', 't1_cciixvk', 't1_ccik36d',
    't1_ccik429', 't1_ccikowu', 't1_ccip0by', 't1_ccj23jt', 't1_cb3h1j9',
    't1_cb3h5qc', 't1_cb43t7z', 't1_cb44gly', 't1_cb3hcuv', 't1_cb3gj80',
    't1_cb3he94', 't1_cb3han6', 't1_cb3hoak', 't1_cb3javy', 't1_cbdfj9b',
    't1_cbdodgh', 't1_cug21tv', 't1_cbsfdbt', 't1_cbsfdsh', 't1_cbsfzmb',
    't1_cbt1zp5', 't3_29e02r', 't1_cik0nb3', 't1_cik10zh', 't1_cikfpnv',
    't1_cikl7o2', 't1_ciko33h', 't1_cil08vt', 't1_cikojhb', 't1_cikoooq',
    't1_cikot2f', 't1_ciksz7j', 't1_ciwiglt', 't1_cikbfes', 't1_cikb6zn',
    't1_cikn4ki', 't1_cikn7j2', 't1_cikp0hi', 't1_cikhbhg', 't1_cikhhoa',
    't1_cikhmdh', 't3_2b3p94', 't1_cj1jb17', 't1_cj1jfd0', 't1_cj1lhpg',
    't1_cj1mlgb', 't1_cj1plx1', 't1_cj1qf2l', 't1_cj1ta76', 't1_cj1taks',
    't1_cj20rt3', 't3_1cb3pd', 't1_c9erwp8', 't1_c9f4p0c', 't1_c9etprq',
    't3_1cd5dm', 't1_c9fcqwc', 't1_c9fdjt4', 't1_cqdsvuv', 't1_cqdsobc',
    't1_cqdtvt5', 't1_cqdvgx3', 't1_cqevg02', 't1_c9yd1j0', 't1_c9yd9d8',
    't1_c9ydmcp', 't1_c9ydp3b', 't1_c9ydsv7', 't1_c9ydv9t', 't1_c9ye0ac',
    't1_c9ye1t1', 't1_c9ye3mn', 't1_cnczw88', 't1_cpvw146', 't1_cpwaiv3',
    't3_2ho8k5', 't1_ckuh8hs', 't1_ckuqadl', 't1_ckuqi1g', 't1_ckuqn0l',
    't1_ckuqtwr', 't1_ckurb23', 't1_ckvjrwg', 't1_ckuwqi8', 't1_cu2w6p8',
    't1_cu2v4xr', 't1_cu3b0fe', 't3_1ybhb0', 't1_cjnx0sk', 't1_cjo1q0q',
    't1_c9v8fs2', 't1_c9vlru6', 't1_c9vihnq', 't1_c9vj17l', 't1_c9v7356',
    't1_c9v7uuu', 't1_c9vn7p4', 't1_c9vo0sl', 't3_1e63c7', 't1_c9xajdg',
    't3_1haww9', 't1_casm5v9', 't1_casuq4a', 't1_castzen', 't1_cat7r28',
    't1_cat7qql', 't1_catelpi', 't1_catjx82', 't1_catknzl', 't1_catmvnu',
    't1_casolkw', 't1_casoidy', 't1_casqgem', 't1_casqo9d', 't1_casqt3t',
    't1_casuw6d', 't1_casoqzt', 't1_casqwgm', 't1_casuynv', 't1_casoa5k',
    't1_castiji', 't1_caxueen', 't1_catr9j9', 't1_catv41c', 't1_catvoqn',
    't1_catx9hg', 't1_catxnxu', 't1_caty2hj', 't1_cav2nyk', 't1_caw8joe',
    't1_cay71op', 't1_caybyz3', 't1_caywd53', 't1_casqjde', 't1_casr12u',
    't1_castbcs', 't1_cggd8lf', 't1_cscanwg', 't1_cscavpa', 't1_csccp8p',
    't1_cscbr0t', 't1_cscbxg2', 't1_cscf2qw', 't1_cscg2p4', 't3_242qn0',
    't1_ch33462', 't1_ch33aph', 't1_ch33yi9', 't1_ch30sac', 't1_ch32zj8',
    't1_ch36a9d', 't1_ch3gnvh', 't1_ch3kgyz', 't1_ch3m71x', 't1_ch3p1te',
    't1_ch33rrs', 't1_ch33u0t', 't1_ch30u5o', 't1_ch3fpaz', 't1_ch3j0r0',
    't1_ch32wl6', 't1_ch34vso', 't1_ch338wb', 't3_1t7zw6', 't1_ce5jtin',
    't3_21yr8u', 't1_cghq6uu', 't3_302atl', 't1_cpojaxi', 't1_cpoores',
    't1_cpp7ghc', 't1_cpra9u0', 't1_cpom9oq', 't1_cpozwy5', 't1_cpoluuw',
    't1_cpovurm', 't1_cpon0c6', 't1_cpotcwb', 't1_cpp3cnv', 't1_cpp5zkb',
    't1_cqcq11x', 't1_cpomrk6', 't1_cponany', 't1_cpov03f', 't1_cpp08sy',
    't1_cpp4871', 't1_c9r3p01', 't1_c9qxy3z', 't3_1d3ijz', 't1_c9mqknk',
    't1_c9na2tf', 't1_ch7cvu3', 't1_ch7ddwy', 't1_cai3dw5', 't1_cai5p4w',
    't1_caidubx', 't3_1eo2kq', 't1_ca23e0m', 't1_ca24dau', 't1_ca24zer',
    't1_ca27qym', 't1_cs4z5zw', 't1_caumnu6', 't1_caunbp5', 't1_cauphni',
    't1_caupmcb', 't1_cauq8sg', 't1_cauqd3a', 't1_caurhj9', 't1_cautyr8',
    't1_cauukxk', 't1_cauutzf', 't1_cauqajh', 't1_cauqglb', 't1_cauqht4',
    't1_caurlql', 't1_cauuln3', 't1_caut5rf', 't1_cav4ico', 't1_cavz2bt',
    't1_cauryqf', 't1_cauqlq8', 't1_cav3xqq', 't1_cauu6uz', 't1_cauqo1h',
    't1_cauqxex', 't1_cavjf1d', 't1_caus15f', 't1_cauyb6p', 't1_cav1xf3',
    't1_cav31oj', 't1_cauppc6', 't1_cayikrk', 't1_cav7n9h', 't1_cav6fuw',
    't1_cav7ncj', 't1_cav271o', 't1_cauv5xm', 't1_cavjcvk', 't1_cavna3e',
    't1_cavorrb', 't1_cavyr72', 't1_cavypo6', 't1_cb1xihc', 't1_cb26x35',
    't1_caxx3kk', 't1_cb1amc8', 't1_cb1j4c4', 't1_cb1jo2k', 't1_cb1kpaq',
    't1_cb3qgr4', 't3_228jbo', 't1_cgke088', 't1_cgke5cw', 't3_28o7uj',
    't1_cicu5de', 't1_cicu6hc', 't1_cicwthx', 't1_cid0cpv', 't3_1fyfb5',
    't3_1fmiv1', 't1_cabqgjf', 't1_cddmfth', 't3_1ibg1p', 't1_cb2zhvu',
    't1_cb3ecr8', 't1_cb32ac8', 't1_cb3grrv', 't3_1v8eur', 't1_cepre6f',
    't1_ceprjui', 't1_ceprovz', 't1_cepsudn', 't1_cepx3y0', 't1_ceqx69p',
    't1_ceqzwux', 't1_cer0f8n', 't1_cer1mzk', 't1_cer1v34', 't1_cer1o8f',
    't1_cer1win', 't1_cer9bme', 't1_cer1d46', 't1_cer1xlw', 't1_cer3x2w',
    't1_cer49ut', 't3_1kvsyr', 't1_cbt4gif', 't1_cbti486', 't1_cbujuoi',
    't1_c8sl4p4', 't1_c8t6pyq', 't1_chjfhp9', 't1_chjgd0q', 't1_chjlkq1',
    't1_chjoacg', 't1_chjp9ke', 't1_chjq0kg', 't1_chlsagj', 't3_2vky2k',
    't1_coikt14', 't1_coinwc1', 't1_coj0szh', 't1_cokopqu', 't1_coin2a8',
    't1_coinuey', 't1_coixxlu', 't1_cgxdmh3', 't1_cgxdsgy', 't1_cgxqk31',
    't1_cgxt74l', 't1_cgxtaif', 't1_cgxlked', 't1_cgxt9jf', 't1_cau88hc',
    't1_cau8ovv', 't1_cau9jjt', 't1_cau8emg', 't1_cau9bgr', 't1_cauamra',
    't1_cau9452', 't1_cauatj3', 't1_cau955t', 't1_cau9nhu', 't1_cauc7tb',
    't1_caue431', 't1_caubzza', 't1_cauchlr', 't1_caueou3', 't1_cauo55g',
    't1_cauhm6v', 't1_cauxek7', 't1_cauhuox', 't1_caunrrz', 't1_caupvos',
    't1_cav04jm', 't3_2eyfoa', 't1_ck4715t', 't1_ck473q2', 't1_ck47es2',
    't1_ck496qz', 't1_ck498fu', 't1_ck46us6', 't1_ck47ebr', 't1_ck4kq73',
    't1_ck4jqhn', 't1_chk6r9i', 't1_chk6s8c', 't1_chk7ilh', 't1_chkg1f1',
    't1_chk8gs0', 't1_chke0un', 't1_chk8ser', 't1_chkfbo3', 't1_chkxjvq',
    't1_chkbwxl', 't1_cl696s0', 't1_cl69kyx', 't1_cl6a7as', 't1_cl6wrm4',
    't1_cl6xdwv', 't1_cl88xn5', 't1_cl89ogw', 't1_cl89vmf', 't1_cl8a4kq',
    't1_cl6aecx', 't1_cbogk1v', 't1_cbognd1', 't1_cboiu19', 't1_cbomaor',
    't1_cbogq70', 't1_cboi8d2', 't1_cboisgm', 't1_cboh71d', 't1_cbp7hcg'
]

def remove_quotes(text: str) -> str:
    """
    Remove quoted lines starting with '>' or '&gt;'.
    """
    return re.sub(r'(?m)^\s*(?:&gt;|>).*?(?:\n\n|$)', '', text).strip()


def find_comment_by_name(comments: list, target_name: str) -> dict:
    for comment in comments:
        if comment.get("name") == target_name:
            return comment
        # unify children → replies
        if "children" in comment:
            comment["replies"] = comment.pop("children")
        for reply in comment.get("replies", []):
            found = find_comment_by_name([reply], target_name)
            if found:
                return found
    return None


def init_persuasion(comment: dict) -> dict:
    comment["persuasion_success"] = 0
    if "children" in comment:
        comment["replies"] = comment.pop("children")
    for r in comment.get("replies", []):
        init_persuasion(r)
    return comment


def process_comment(comment: dict, all_comments: list) -> dict:
    comment["body"] = remove_quotes(comment.get("body", ""))
    if "children" in comment:
        comment["replies"] = comment.pop("children")

    new_replies = []
    for reply in comment.get("replies", []):
        # detect persuasion delta
        if reply.get("author") == "DeltaBot" and "delta awarded" in reply.get("body", "").lower():
            m = re.search(r"/u/(\w+)", reply["body"])
            if m:
                target = m.group(1)
                parent = find_comment_by_name(all_comments, reply.get("parent_id"))
                if parent:
                    grand = find_comment_by_name(all_comments, parent.get("parent_id"))
                    if grand and grand.get("author") == target:
                        grand["persuasion_success"] = 1
        else:
            new_replies.append(process_comment(reply, all_comments))

    out = {
        "name": comment.get("name", "unk"),
        "author": comment.get("author"),
        "created_utc": comment.get("retrieved_on", -1),
        "body": comment.get("body", ""),
        "persuasion_success": comment.get("persuasion_success", 0),
        "parent_id": comment.get("parent_id"),
        "replies": new_replies
    }
    return out


def collect_rows_from_comments(comments: list, rows: list) -> None:
    for c in comments:
        if c["name"] not in legal_name_list:
            rows.append({
                "name": c["name"],
                "author": c["author"],
                "created_utc": c["created_utc"],
                "body": c["body"],
                "persuasion_success": c["persuasion_success"],
                "parent_id": c["parent_id"]
            })
        for r in c.get("replies", []):
            collect_rows_from_comments([r], rows)


def produce_csv_rows(threads_path: str) -> list:
    rows = []
    with open(threads_path, 'r', encoding='utf-8') as f:
        for line in f:
            thread = json.loads(line)
            # init
            thread_comments = [init_persuasion(c) for c in thread.get("comments", [])]
            processed = [process_comment(c, thread_comments) for c in thread_comments]

            # add post as row
            title = thread.get('title', '')
            body = thread.get('body', '')
            post_body = remove_quotes(title + '\n\n' + body)
            post = {
                'name': thread.get('name', 'unk'),
                'author': thread.get('author'),
                'created_utc': thread.get('created_utc'),
                'body': post_body,
                'persuasion_success': 0,
                'parent_id': None
            }
            if post['name'] not in legal_name_list:
                rows.append(post)
            # comments
            collect_rows_from_comments(processed, rows)
    return rows

# --- Step 2 + 3: In-memory JSON conversion and limiting discussions ---

def limit_discussions(items: list, num_discussions: int) -> list:
    # map by id
    id_map = {i['name']: i for i in items}
    # build children map
    children = defaultdict(list)
    for i in items:
        pid = i.get('parent_id')
        if pid and pid in id_map:
            children[pid].append(i['name'])

    def recurse_collect(root_id):
        out = [id_map[root_id]]
        for cid in children.get(root_id, []):
            out.extend(recurse_collect(cid))
        return out

    result = []
    count = 0
    for i in items:
        if i['name'].startswith('t3_'):
            if count >= num_discussions:
                break
            result.extend(recurse_collect(i['name']))
            count += 1
    return result

# --- Main pipeline ---

def main():
    parser = argparse.ArgumentParser("Pipeline: threads.jsonl → CSV → JSON → limited discussions")
    parser.add_argument('threads_file', help='Input threads.jsonl')
    parser.add_argument('output_json', help='Final output JSON path')
    parser.add_argument('--num-discussions', type=int, default=5,
                        help='Max number of top-level discussions to include')
    args = parser.parse_args()

    # Step 1: CSV rows
    print("Generating CSV-like rows from threads...")
    csv_rows = produce_csv_rows(args.threads_file)

    # Step 2: (Already in dict form) → JSON list
    print(f"Total rows: {len(csv_rows)}")
    # Step 3: limit discussions
    print(f"Limiting to {args.num_discussions} discussions...")
    limited = limit_discussions(csv_rows, args.num_discussions)

    # Write final JSON
    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as outf:
        json.dump(limited, outf, ensure_ascii=False, indent=2)
    print(f"Wrote {len(limited)} items to {args.output_json}")

if __name__ == '__main__':
    main()
