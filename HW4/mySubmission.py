#############################################################
# Problem 0: Find base point
def GetCurveParameters():
    # Certicom secp256-k1
    # Hints: https://en.bitcoin.it/wiki/Secp256k1
    _p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    _a = 0x0000000000000000000000000000000000000000000000000000000000000000
    _b = 0x0000000000000000000000000000000000000000000000000000000000000007
    _Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    _Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    _Gz = 0x0000000000000000000000000000000000000000000000000000000000000001
    _n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    _h = 0x01
    return _p, _a, _b, _Gx, _Gy, _Gz, _n, _h

class Point:
    def __init__(self, x=0, y=0):
        self._x = x  # 初始化x座標
        self._y = y  # 初始化y座標

    def x(self):
        return self._x  # 返回x座標

    def y(self):
        return self._y  # 返回y座標

    def set_x(self, a):
        self._x = a  # 設定新的x座標

    def set_y(self, b):
        self._y = b  # 設定新的y座標

def double(G):
    p, a, b, Gx, Gy, Gz, n, h = GetCurveParameters()
    
    m = ((3 * G._x ** 2 + a) * pow(2 * G._y, -1, p)) % p
    x1 = (m**2 - G._x - G._x) % p
    y1 = (m * (G._x - x1) - G._y) % p
    R = Point(x1, y1)
    return R
    
def addition(P, Q):
    p, a, b, Gx, Gy, Gz, n, h = GetCurveParameters()
    if P._x == Q._x:
        return None
    else:
        m = ((Q._y - P._y) * pow(Q._x - P._x, -1, p)) % p
    x1 = (m**2 - P._x - Q._x) % p
    y1 = (m * (P._x - x1) - P._y) % p
    
    R = Point(x1,y1)
    return R    
    
#############################################################
# Problem 1: Evaluate 4G
def compute4G(G, callback_get_INFINITY):
    """Compute 4G"""
    result = callback_get_INFINITY()
    Gx = G.x()
    Gy = G.y()
    G = Point(Gx, Gy)
    P = Point(Gx, Gy)
    P = double(P)
    P = double(P)
    result = P
    """ Your code here """
    return result


#############################################################
# Problem 2: Evaluate 5G
def compute5G(G, callback_get_INFINITY):
    """Compute 5G"""
    result = callback_get_INFINITY()
    Gx = G.x()
    Gy = G.y()
    G = Point(Gx, Gy) #G
    P = Point(Gx, Gy) 
    P = double(P) #2G
    P = double(P) #4G
    P = addition(P, G) #5G
    result = P
    return result


#############################################################
# Problem 3: Evaluate dG
# Problem 4: Double-and-Add algorithm
def double_and_add(n, point, callback_get_INFINITY):
    """Calculate n * point using the Double-and-Add algorithm."""
    Gx = point.x()
    Gy = point.y()
    """ Your code here """
    result = callback_get_INFINITY()
    num_doubles = 0
    num_additions = 0
    G = Point(Gx, Gy) #G
    P = Point(Gx, Gy)
    bi = bin(n)[3:]
    
    for i in bi:
        if i == '0':
            P = double(P)
            num_doubles += 1
        
        elif i == '1':
            P = double(P)
            P = addition(P, G)
            num_doubles += 1
            num_additions += 1
            
    result = P

    return result, num_doubles, num_additions


#############################################################
# Problem 5: Optimized Double-and-Add algorithm
def optimized_double_and_add(n, point, callback_get_INFINITY):
    """Optimized Double-and-Add algorithm that simplifies sequences of consecutive 1's."""

    """ Your code here """
    result = callback_get_INFINITY()
    num_doubles = 0
    num_additions = 0
    Gx = point.x()
    Gy = point.y()
    G = Point(Gx, Gy) #G
    nG = Point(Gx, -Gy) #-G
    P = Point(Gx, Gy) # result point
    bi = bin(n)[3:]
    consecutive_one = 0
    for i in range(len(bi)):
        if i == 0:
            if bi[i] == '1':
                P = double(P)
                num_doubles += 1
                consecutive_one +=1
        else:
            if bi[i] == '1': 
                if bi[i-1] == '0': #01
                    P = addition(P, G)
                    num_additions += 1
                consecutive_one +=1
            else:
                if bi[i-1] == '1': #10
                    if consecutive_one == 1:
                        if i == 1:
                            num_doubles -= 1
                        else:
                            num_additions -= 1   
                    P = addition(P, nG)
                    num_additions += 1
                consecutive_one = 0
        P = double(P)
        num_doubles += 1
    
    if consecutive_one > 0:
        if consecutive_one == 1:
            num_additions -= 1
        P = addition(P, nG)
        num_additions += 1
    
    result = P
    return result, num_doubles, num_additions


#############################################################
# Problem 6: Sign a Bitcoin transaction with a random k and private key d
def sign_transaction(private_key, hashID, callback_getG, callback_get_n, callback_randint):
    """Sign a bitcoin transaction using the private key."""
    """ Your code here """
    G = callback_getG()
    n = callback_get_n()
    Gx = G.x()
    Gy = G.y()
    z = int(hashID[:n], 16)
    k = callback_randint(1, n-1) 
    B = Point(Gx, Gy) #G
    P = Point(Gx, Gy)
    bi = bin(k)[3:]
    for i in bi:
        if i == '0':
            P = double(P)
        else:
            P = double(P)
            P = addition(P, B)
    
    r = P._x % n 
    s = (pow(k, -1, n)*(z + r * private_key)) % n
    signature = (r,s)
    return signature


##############################################################
# Step 7: Verify the digital signature with the public key Q
def verify_signature(public_key, hashID, signature, callback_getG, callback_get_n, callback_get_INFINITY):
    """Verify the digital signature."""

    """ Your code here """
    G = callback_getG()
    n = callback_get_n()
    if public_key == callback_get_INFINITY():
        return False
    if n * public_key != callback_get_INFINITY():
        return False
    
    r = signature[0]
    s = signature[1]
    if not (1 <= r < n and 1 <= s < n):
        return False  
    
    z = int(hashID[:n], 16)
    w = pow(s, -1, n) % n
    u1 = (z * w) % n
    u2 = (r * w) % n
    U1,_,_ = double_and_add(u1, G, callback_get_INFINITY)
    U2,_,_ = double_and_add(u2, public_key, callback_get_INFINITY)
    result = addition(U1,U2)

    return (r == (result._x) % n )

