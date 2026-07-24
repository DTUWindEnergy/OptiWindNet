"""Verified historical aliases for bundled location coordinate digests."""

from __future__ import annotations

from functools import cache
import pickle
from pathlib import Path

import numpy as np

from optiwindnet.fingerprint import fingerprint_coordinates

__all__ = (
    'NODESET_DIGEST_ALIASES',
    'NODESET_DIGEST_ALIAS_COORDINATES_FILE',
    'NODESET_DIGEST_ALIAS_NAMES',
    'nodeset_digest_alias_coordinates',
)

# These database NodeSets differ from current bundled coordinates because of
# projection/runtime drift, but each pair was verified to have matching T/R/B
# metadata and to generate the same A nodes and edges without renumbering.
# Values are canonical repository digests, so aliases for single-root variants
# retain their variant identity. Fecamp is intentionally excluded: its database
# NodeSet requires a permutation of 69 turbine ids before its A edges match.
#
# This is a data table: keep each historical/canonical pair together for review.
# ruff: noqa: E501
# fmt: off
_ALIAS_HEX = {
    '04117f1b5afbb16f22bdb83e6f974393190b3f7ff9d9fd00386b613d9d112876': ('2189cba503362c7ae78b9559fa283245f4b9b37f9fcd9dc0de9f653acbb05204', 'Rampion'),
    '9386a0040730d5d0c6604cb737d723f2943c4862272066eaccf840f65d8b3c13': ('8b0d1fcda8dee5d5c2121c98ae1e59cd97e85e0d30e6f1f2b12fcde574070bc5', 'Hornsea One'),
    'b92178f32869c12125b9f8cb93ae806643fb27c1fa6a9b5c44adc484c7977987': ('7fc88f3c22c7bad91ab8aee75496185430baf7b4f7afdade73fe9e9219f4608c', 'Race Bank'),
    'e6eeff82b004dc4ce76eafd178d7713469b37fe89a81d82532189c7a89a7d277': ('00fe7cb7502c69323d28b9c4c9a2f505ff928360e47ace44cfdd5e6c58284283', 'Horns Rev 1'),
    '8e74e71c8e5fdf7b11ff7e0779345729c0bb9001544d940260c170f2e6473c62': ('e92e376fbf64a781568bcede6c9f75ccbd0a00ee707a9d69735fdfe9bedb3a2e', 'West of Duddon Sands'),
    '54b7ec5f5340b7cd11410ffa17102929c5a1c5b72ae3a955fa6c36406141868d': ('c141db23471367ba8f355525abbea19582d545f927f41809ae0907157a259ebe', 'Greater Gabbard Inner'),
    '1ed8a88c02f76b4fe88cd49ce53fc6d4bb747807fb1323ce9bd5a96fa0cbc957': ('50b61c083b479cabbc4bb31d0404a746134bc7b83a1e324853889223af5ba182', 'Borkum Riffgrund 1'),
    '67d27baa4b2cfdd96dc5789a8d14b2654959674ac5faa985b635e179f9325a80': ('6ca49a38c0c466516c022b566cf4256262b88eee8c111214bd254a71061985f6', 'Anholt'),
    '86f4ff7dcb42c31dba1fdf52f3b96b13989945217b0899c01262efa775fba73b': ('3642b923c352e0a9579851677607ffb9ff7bba0456986376450899d4503e74ba', 'Gwynt y Mor'),
    '23180ca23f1973d3f3d63da2cd3186b2376a9a20bea3337c83115f43d49e16b3': ('4e9c9940f6868b0f348662a50d284c2f3d4351510954d5c7a226766bcb0762a6', 'Triton Knoll'),
    '0031546bb546ee129cfe9646ba8355b7787745cab498a4aca7ce485c6676ca9f': ('69d1cdd1f3c32f18bd6cde2f5977b41b5204414c588d7d1da40dae98033c51fa', 'Thanet'),
    'bbd4cfb0972da0ae94e204a301f4bc7f4c06d5bf3b1c9050f7c06cf461b1c821': ('f01d761ac8ecf4e29e83567094f97052907edd0fe629969563f127c3bb81c74d', 'London Array'),
    '57ff60cd6254b787353130df96e22f5980a3c8936f27a5d7352860e1f3f544c5': ('beba882bdb0b53a2a9547e6bccd0d23df062bc4577e880192aa21b58f9b58b92', 'Rødsand 2'),
    'c77b1d567dd36b270a38018750c3546542aa1bca8f90f00306c2b90a9cee85ff': ('86c2770735326100b543351b7f5a2859a02ad0ddaa371f187e5f2ee682bebcce', 'Meerwind'),
    '25bfffd1d3c47f081aa1374c39108c7d59c55d6c3b371a596c07cb3718c61e81': ('d551cd458eac8622c595ced94089b6c2179c141f72a21681ab9f7bbe25b8a5f8', 'Dogger Bank C'),
    '230a6b92da72723b6db3d42a8ebbec42ea1156bedd5fc7584a2948347345eb7a': ('79fe1a164f64cc9f3e7531494786163b7ae13e29951ab5821a85ed863ba52f27', 'Northwind'),
    'f60bebda8e899af586844e8dbaa7529249c00334645e7f74b343fce218f85509': ('fb70c98584b6b2239b93706f25aace856e66c6eee0a9c44277fad503b10e5db9', 'Gemini 1'),
    '8c80457e16a0ea02fb21f8d8df73be8f84f571e3d964b4620bce0a87cab04410': ('4b7a0d6cd12e0cc3434052a0a4838a5e773d671dc7ffc090af3c1f1401771b53', 'Galloper Inner'),
    '2bc88f77402b5f2168ca57641b7cba2d7b4b6ac1ece83642317f587852990040': ('d749569e274072589e0d3d3a9c373a09baf192b36a3979ff07d535da41d8e667', 'Rudong H8'),
    'e19d6004bc4f26ec6bab01dd5ddbbfe0e5c36e8d9aecdbe800f8642a476545eb': ('eb051375746d7e48d873104bed629443cfa0b680f4b863d2fece62e3a3c1ad74', 'Rentel'),
    '3977dc27ca5ac96e0438f4f2d0407b6c7d924a667d7c400414deb133cd6aa842': ('d8b61d8fc884e5a851c2990b8f41b22af098f203c3947631afd1bef0fef79ea4', 'Merkur'),
    '0f3320abc7c384b1814b072a32b56b2b79f2342f33d3f18796b4b8a2f50a4f5b': ('d637a069759b499d410b7ef44d44c34b4413ea0783a6a91473687fc574cf57d9', 'Riffgat'),
    'f1ed6661cc5fd2211c378d30bb531e84bd004d2362b1022417a32d6fe8d4d49d': ('32eec1c7d0cf7fa02d9588ee59fbd1d1855edd0bfe6fa0eff3073cff1542b9ba', 'CECEP Yangjiang Nanpeng Island'),
    'aec1d812f5e60120e34612110a129431455501c1806284289413610091b5303e': ('73efd449a41876b276a57dcb6b79462103b1f3015f75e73cf8f408558746724b', 'Saint-Brieuc'),
    '70689116e7326ce7e4a38524533b62512d84229f77f8b64b2c07e992958ab3a7': ('3709c385eec67a0bbf95154e7c29186a130ec83d638c1d63c63603f9d238bc53', 'Lincs'),
    '8e1447881397d27eda54d920bd83e52bbbe5d040f8a2a6c30838ba4f58268e90': ('71836f55ae6dcc6fa05088ca3fe1333244be620c45959eadb571fa1783b71502', 'Humber Gateway'),
    '2bc68dc1cc8a462cc4f869f19eebd8e1c895082b8b85b2947334cbf56d721ff0': ('c11c749cc6903d95c8e83644fe2710f552348afde93e834481a0e071fd1bd40e', 'Saint-Nazaire'),
    '799752b96a75edcd15e5ebdc7daecba5ce3ef87a8463d7d9fe3abd65f4b247b2': ('171ef2fe0df2566515f93a1c64c5392e0a987044b78addf8a4eee2dbe39f8a6e', 'CGN Rudong Demonstration'),
    'a09f50a5e18e50df9d2816419842afd8f2895960e2d574375e765112f0c43cc8': ('0d605a54c5daf933c74265cf45f56fa73cb48c1d00b359cb267da45233529222', 'Gemini 2'),
    'aef77d89addf128e02c33b92f43d6925f8b98c272991bdd684d2f3af207007c8': ('7aa94919c706b224efe1fc67f4f3fd4bfe506521177e5fa828e1731a06dc9887', 'Deutsche Bucht'),
    'e6a5c8ce2bf320e950b98a96420a1fbcb2ed158621ea183751136bf1b9aec872': ('fd9e03dca1501d8f7e4dfc506c107c70b05dd7186a36dce9d51ebf4a1e0d5c25', 'Kaskasi'),
    '387b19cf0a536227fb74e3ac0b816d338a79c87fff346f813fbbf0fd939432ab': ('c974c670406cb2d63c8fabff929ff7c5b1c6407b6ba035593038a30953e9eee3', 'BARD Offshore 1'),
    '43c864df73720384e650e839feb71fff4e43f1ce2bf8bf01ed55477c9423d532': ('8babe1526742de26ccc9ae78db49c2f72baf57b64b3c0ecd7ddab299e3f7466e', 'Nordsee Ost'),
    'e62bbe1a01a2ba27c7239a2539415043224dbe4c4cc36743cd1dcefafa551efc': ('54e7afc75cbca705197da371183d07532efa9c00df84eb65f80d33aa356616cf', 'Luchterduinen'),
    '325f704e6e1f392fd94f42cfb0f81154842ba0da9b77184a7fa74f074292ca74': ('6664658a0359bf513f189b58fd88d2a0d33ed6303c4d2a40974d1f9a649f4629', 'Princess Amalia'),
    'c9e2c7a50ec3d81f5651738f146299ec7a8922dc71791b553a57ea2ca7984ecb': ('46e1ce20858cfbb27228472975f18065da1133dd251e2e0b654ad44e7bb9ecdd', 'Norther'),
    '30d0627357a58aeb02237c7dae3803d681c62c3503de9444b07ba6b15eb3d83a': ('c807f1efc1a2035cc3ceef2bf32a76795b4d11cf773a0fe1cb80750525e20e78', 'Hohe See'),
    'ba7e63eaebb4a695b122cf18aabb5bc5c5909751d1b7e3fc4cf20c3edb008a1c': ('e765548f0af25cc36cfcdf60c6ee9cf30623697cbd05f40dae7df5e8e851f506', 'Nordsee One'),
    'dd9a2e65086bf1a23e040a7d5c1af01a93dda6e60011ed7b2a373e178eaf59f2': ('a97e81680421927841491370d4460ad09e49ca4316ed9f1ffdca496033780c8b', 'Baltic Eagle'),
    'd162ed9df1eb7fb49d1af5f81823adf6382d77f7f21af64870258915c5fb67d6': ('301ab40e385e0067535ceeb6feb19571908d1e204b9759d5f90896c83e1f4c65', 'Inch Cape'),
    '9d0c17fa3988985402724053e4e0a2ffdf400c1588434e78af1a046feb4bff2e': ('4239216ccf41c2f1eaca0d533f849600ce53fea54a4ac7e0aefc3357b56de3f2', 'Westermost Rough'),
    '4583fc69268aa352f72dfd2b75411db6db0f00eaf6f09ae1cffe8acbcf12fd90': ('b0f5c79382781d1de04b2ba44cb5d0379b7f0b5cf2c423f8109efc9dfe1d5ecf', 'SPIC Binhai North H2'),
    '525ac816f897fcedb1ba162578f02a35723588c10ba1ffe6552f7829c666d790': ('932a7a517318019bf19045b9f28eeac437564a5739a19e804a2016bcfea3a27d', 'Rudong H10'),
    '9cfec3bc60d49abe93faa16665b7d7afbc7ebdf7dff01fc88f2dafc26e4309ca': ('f5cf89969207dd69d9bb48c38e1d823ca78e9380480a24e0d87a9edbdd69f230', 'Trianel Windpark Borkum'),
    '3f967129527c2a7b753dcb2c49f469498fffa8bb24fb409602061a8c917d0b61': ('b89f57c048cfc37dd63ae7b37bf2aea157f2fbac1a06720825a391380527bf2d', 'Amrumbank West'),
    '8bce7665571e03edc46f37df038a5981002b33f94c4301a866470d5bc76e62d7': ('419aad1063a07b57e1755826e916bbdad66dac68cd87d8cda0c9ba176d69bc53', 'Lillgrund'),
    '1f073d2f8112951f2872246fee3de1a324f18e92eaf373eadf1017cd0f7bb171': ('dc2e44e76f0016a8f3c2dcf06f0822a64b3a254c6ef371e43f45aadf65e50ab3', 'Kriegers Flak B'),
    '90f5a090394bd131d54062e90d1621882d5298bb5a2c5dd551f7cbddbeb94c2a': ('48aaa767bbe5880b50b49963b0be204fca1636c0278c4265a5dfb66011c94e62', 'Belwind'),
    '38d87d16e95991384396b0883c26e76695c5e660fcdeea752506916cdae43c18': ('da268de0daf522344fc700bf6da26556ba1c77d0a960dce9d89b9d2dfab38be4', 'Thor'),
    '2ed1c2f81248e2f515aeca81d61500017368e4df5b1d85e80d02d786b19c5562': ('8acec60c2cf716ef39ad24d7d519c8d7e764c5679a8b2f4272fb728061ade3f0', 'Global Tech 1'),
    'de727817741bf297f401779da9d1cffbb758eafb98fd2afd29cc9a18a02157c8': ('0cb221075f31637339ec44f17145859b1c8b78904b6e651bac2ade15339cd5bf', 'Albatros'),
    'abfbcbf16da7dc3050cc422168c94f7b9a796d7373c324000d08f0099f6dd418': ('ff8af0a59bc5098aad357a8f2db0d8c5c83ceeb2e316fe4a7bd74b264d306cfb', 'Veja Mate'),
    'ec4334fdc2f4995062c442848410bf7163a3d8e60b1dab3d91faddfd79717118': ('5419e21473cf79abd8286e384d67eb0e1478b19d928633ab7c001e4f8559d39f', 'Kriegers Flak A'),
    'bb146f66e021553349486736f98f63aac67d19854a134d2cbd946790efeca189': ('d86d9a5d1b981f90aa180b796bfc0dd5ab6c7481ab8a36a7a7ffb70d6d9ae9dd', 'Mermaid'),
    'a5232ada2454bc65cecb138c7db49d4df6722b50862c31bca30713bb47c695ce': ('40232fd051066ef0c220d6c8ed89d28ac102c9b546e73775a59fa91c37cb6314', 'Seagreen'),
    '58c89806ee5a0f7a6f16fc39ed50734edf815dde27779f49a59e07b3e698f394': ('915d172db7bd202f2c2e50429b3263dc16d891ea7b01ee0a015acd7965d33653', 'Neart na Gaoithe.1_OSS'),
    '3c456ef308686e2b71031f8cee4ab3364e1ea3c167713d3de0d758ee1ca4fbf8': ('0696a251a99def7c4c8521f5453a9e46bd4d9dc98d21c2ed51587f7b6c4cd6ef', 'Sheringham Shoal.1_OSS'),
    '8a2672a6c7af6b4f1e83915514e875347adac723710015728109d1913e6b6974': ('504c3bbd51978cbe03123bc31a35691879ca6f99c86675f10d077b4e74ee0520', 'Beatrice.1_OSS'),
}
# fmt: on

NODESET_DIGEST_ALIASES = {
    bytes.fromhex(historical): bytes.fromhex(canonical)
    for historical, (canonical, _name) in _ALIAS_HEX.items()
}

NODESET_DIGEST_ALIAS_NAMES = {
    bytes.fromhex(historical): name
    for historical, (_canonical, name) in _ALIAS_HEX.items()
}

NODESET_DIGEST_ALIAS_COORDINATES_FILE = Path(__file__).with_name(
    'nodeset_digest-alias-coordinates.pkl'
)


@cache
def nodeset_digest_alias_coordinates() -> dict[bytes, np.ndarray]:
    """Load and validate historical coordinates for every digest alias."""
    if not NODESET_DIGEST_ALIAS_COORDINATES_FILE.exists():
        raise FileNotFoundError(
            f'Missing node-set alias coordinates: '
            f'{NODESET_DIGEST_ALIAS_COORDINATES_FILE}\n'
            'Regenerate them with: '
            'python -m tests.update_nodeset_digest_aliases <routesets.sqlite>'
        )
    with NODESET_DIGEST_ALIAS_COORDINATES_FILE.open('rb') as file:
        coordinates = pickle.load(file)
    if not isinstance(coordinates, dict) or set(coordinates) != set(
        NODESET_DIGEST_ALIASES
    ):
        raise TypeError('node-set alias coordinates do not match the alias table')
    for digest, vertexc in coordinates.items():
        if (
            not isinstance(vertexc, np.ndarray)
            or vertexc.ndim != 2
            or vertexc.shape[1] != 2
            or fingerprint_coordinates(vertexc)[0] != digest
        ):
            raise TypeError(f'invalid coordinates for node-set alias {digest.hex()}')
    return coordinates
