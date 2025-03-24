ht_default= {'dc0':'00010501010101010100000000000000000102030405060708090a0b',
             'ac0':'0002010303020403050504040000017d010203000411051221314106'\
                   '13516107227114328191a1082342b1c11552d1f02433627282090a16'\
                   '1718191a25262728292a3435363738393a434445464748494a535455'\
                   '565758595a636465666768696a737475767778797a83848586878889'\
                   '8a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9ba'\
                   'c2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9ea'\
                   'f1f2f3f4f5f6f7f8f9fa',         
              'dc1':'00030101010101010101010000000000000102030405060708090a0b',
              'ac1':'00020102040403040705040400010277000102031104052131061241'\
                    '510761711322328108144291a1b1c109233352f0156272d10a162434'\
                    'e125f11718191a262728292a35363738393a434445464748494a5354'\
                    '55565758595a636465666768696a737475767778797a828384858687'\
                    '88898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8'\
                    'b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9'\
                    'eaf2f3f4f5f6f7f8f9fa'       
            }

def buildHT(huf_tables,param='encode'):  # biuld huffman table from Hex-digits
    HT=[]
    for ht in ['dc0','ac0','dc1','ac1']:
        dht=bytes.fromhex(huf_tables[ht]) 
        table = {}
        num_codes_by_length=list(dht[:16])
        code_ptr = 16
        code_val = 0b0
        for code_length, num_codes in enumerate(num_codes_by_length, 1):
            if num_codes != 0:
                for _ in range(num_codes):
                    if param=='decode': # for decode
                        table.update({'{:0{}b}'.format(code_val, code_length):dht[code_ptr]})
                    else:                # for encode
                        table.update({dht[code_ptr]:'{:0{}b}'.format(code_val, code_length)})
                    code_ptr += 1
                    code_val += 1
            code_val <<= 1
        HT.append(table)
    return HT