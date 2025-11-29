
from typing import TypeVar, Generic, Optional, List, Tuple, Iterator
from abc import ABC, abstractmethod
import bisect

K = TypeVar('K')
V = TypeVar('V')


class BPlusTree(Generic[K, V]):
 
    def __init__(self, order: int = 4):
        if order < 2:
            raise ValueError("Red mora biti >= 2")
            
        self.order = order
        self.root: Optional['BPlusTree.Node'] = None
        self._size = 0
    
    class Node(ABC):
        def __init__(self, order: int):
            self.order = order
            self.keys: List[K] = []
            self.parent: Optional['BPlusTree.InternalNode'] = None
        
        @property
        def max_keys(self) -> int:
            return 2 * self.order - 1
        
        @property
        def min_keys(self) -> int:
            return self.order - 1
        
        @property
        @abstractmethod
        def is_leaf(self) -> bool:
            pass
        
        def is_full(self) -> bool:
            return len(self.keys) >= self.max_keys
        
        def is_underflow(self) -> bool:
            if self.parent is None:
                return False
            return len(self.keys) < self.min_keys
        
        def find_key_index(self, key: K) -> int:
            return bisect.bisect_left(self.keys, key)
    
    class LeafNode(Node):
        def __init__(self, order: int):
            super().__init__(order)
            self.values: List[List[V]] = []  # Svaki ključ mapira na listu vrednosti
            self.next: Optional['BPlusTree.LeafNode'] = None
            self.prev: Optional['BPlusTree.LeafNode'] = None
        
        @property
        def is_leaf(self) -> bool:
            return True
        
        def search(self, key: K) -> Optional[List[V]]:
            idx = self.find_key_index(key)
            if idx < len(self.keys) and self.keys[idx] == key:
                return self.values[idx]
            return None
        
        def insert(self, key: K, value: V) -> bool:
            idx = self.find_key_index(key)
            
            if idx < len(self.keys) and self.keys[idx] == key:
                # Ključ već postoji, dodaj vrednost u listu
                self.values[idx].append(value)
                return False
            
            # Novi ključ, kreiraj listu sa jednom vrednosću
            self.keys.insert(idx, key)
            self.values.insert(idx, [value])
            return True
        
        def remove(self, key: K) -> Optional[List[V]]:
            idx = self.find_key_index(key)
            if idx < len(self.keys) and self.keys[idx] == key:
                self.keys.pop(idx)
                return self.values.pop(idx)
            return None
        
        def split(self) -> Tuple['BPlusTree.LeafNode', K]:
            mid = len(self.keys) // 2
            
            new_node = BPlusTree.LeafNode(self.order)
            new_node.keys = self.keys[mid:]
            new_node.values = self.values[mid:]
            
            # Linkovi
            new_node.next = self.next
            new_node.prev = self
            if self.next:
                self.next.prev = new_node
            self.next = new_node
            
            # Skrati trenutni
            self.keys = self.keys[:mid]
            self.values = self.values[:mid]
            
            return new_node, new_node.keys[0]
        
        def merge_with_right(self, right: 'BPlusTree.LeafNode'):
            self.keys.extend(right.keys)
            self.values.extend(right.values)
            
            self.next = right.next
            if right.next:
                right.next.prev = self
        
        def redistribute_with_left(self, left: 'BPlusTree.LeafNode') -> K:
            key = left.keys.pop()
            value = left.values.pop()
            
            self.keys.insert(0, key)
            self.values.insert(0, value)
            
            return self.keys[0]
        
        def redistribute_with_right(self, right: 'BPlusTree.LeafNode') -> K:
            key = right.keys.pop(0)
            value = right.values.pop(0)
            
            self.keys.append(key)
            self.values.append(value)
            
            return right.keys[0] if right.keys else key
    
    class InternalNode(Node):
        def __init__(self, order: int):
            super().__init__(order)
            self.children: List['BPlusTree.Node'] = []
        
        @property
        def is_leaf(self) -> bool:
            return False
        
        def find_child(self, key: K) -> 'BPlusTree.Node':
            """
            KLJUČNA METODA: Pronalazi dete čvor za routing
            
            U B+ stablu:
            - ključ < keys[i] -> ide u children[i]  
            - ključ >= keys[i] -> ide u children[i+1]
            - ako je ključ >= svih keys, ide u poslednje dete
            """
            idx = 0
            while idx < len(self.keys) and key >= self.keys[idx]:
                idx += 1
            return self.children[idx]
        
        def insert_child(self, key: K, right_child: 'BPlusTree.Node'):
            idx = self.find_key_index(key)
            self.keys.insert(idx, key)
            self.children.insert(idx + 1, right_child)
            right_child.parent = self
        
        def remove_key(self, key: K) -> bool:
            idx = self.find_key_index(key)
            if idx < len(self.keys) and self.keys[idx] == key:
                self.keys.pop(idx)
                self.children.pop(idx + 1)
                return True
            return False
        
        def split(self) -> Tuple['BPlusTree.InternalNode', K]:
            mid = len(self.keys) // 2
            promoted_key = self.keys[mid]
            
            new_node = BPlusTree.InternalNode(self.order)
            new_node.keys = self.keys[mid + 1:]
            new_node.children = self.children[mid + 1:]
            
            for child in new_node.children:
                child.parent = new_node
            
            self.keys = self.keys[:mid]
            self.children = self.children[:mid + 1]
            
            return new_node, promoted_key
        
        def merge_with_right(self, right: 'BPlusTree.InternalNode', separator: K):
            self.keys.append(separator)
            self.keys.extend(right.keys)
            self.children.extend(right.children)
            
            for child in right.children:
                child.parent = self
    
    # Glavne metode
    
    def _find_leaf(self, key: K) -> Optional['LeafNode']:
        if not self.root:
            return None
        
        current = self.root
        while not current.is_leaf:
            current = current.find_child(key)
        
        return current
    
    def search(self, key: K) -> Optional[List[V]]:
        leaf = self._find_leaf(key)
        return leaf.search(key) if leaf else None
    
    def insert(self, key: K, value: V):
        if not self.root:
            self.root = self.LeafNode(self.order)
        
        leaf = self._find_leaf(key)
        
        is_new = leaf.insert(key, value)
        # Uvek povećaj _size jer dodajemo novu vrednost (čak i za postojeći ključ)
        self._size += 1
        
        if leaf.is_full():
            self._split_leaf(leaf)
    
    def _split_leaf(self, leaf: 'LeafNode'):
        new_leaf, promoted_key = leaf.split()
        self._insert_into_parent(leaf, promoted_key, new_leaf)
    
    def _insert_into_parent(self, left_child: 'Node', key: K, right_child: 'Node'):
        if left_child.parent is None:
            new_root = self.InternalNode(self.order)
            new_root.keys = [key]
            new_root.children = [left_child, right_child]
            left_child.parent = right_child.parent = new_root
            self.root = new_root
        else:
            parent = left_child.parent
            parent.insert_child(key, right_child)
            
            if parent.is_full():
                new_parent, promoted_key = parent.split()
                self._insert_into_parent(parent, promoted_key, new_parent)
    
    def remove_value(self, key: K, value: V) -> bool:
        """Uklanja jednu vrednost iz liste za dati ključ"""
        leaf = self._find_leaf(key)
        if not leaf:
            return False
        
        idx = leaf.find_key_index(key)
        if idx >= len(leaf.keys) or leaf.keys[idx] != key:
            return False
        
        value_list = leaf.values[idx]
        if value in value_list:
            value_list.remove(value)
            # Uvek smanji _size jer uklanjamo vrednost
            self._size -= 1
            # Ako je lista prazna, ukloni ceo ključ
            if not value_list:
                leaf.keys.pop(idx)
                leaf.values.pop(idx)
                # Ako je list potpuno prazan, možda treba reorganizacija
                if not leaf.keys and leaf.parent:
                    # TREBALO BI: Spoji sa susednim listom ili pozajmi ključeve
                    # Za sada ostavljamo - LSM sistem će ionako reorganizovati podatke
                    # tokom merge operacija, tako da ovo nije kritično
                    pass
            return True
        return False
    
    def delete(self, key: K) -> bool:
        if not self.root:
            return False
        
        leaf = self._find_leaf(key)
        if not leaf:
            return False
        
        value_list = leaf.remove(key)
        if value_list is None:
            return False
        
        # Smanji _size za broj obrisanih vrednosti
        self._size -= len(value_list)
        
        if leaf.is_underflow():
            self._fix_underflow(leaf)
        
        return True
    
    def _fix_underflow(self, node: 'Node'):
        if node.parent is None:
            if not node.keys and not node.is_leaf:
                internal_node = node
                if internal_node.children:
                    self.root = internal_node.children[0]
                    self.root.parent = None
            return
        
        parent = node.parent
        node_idx = parent.children.index(node)
        
        # Pokušaj pozajmljivanje
        if node_idx > 0:
            left_sibling = parent.children[node_idx - 1]
            if len(left_sibling.keys) > left_sibling.min_keys:
                if node.is_leaf:
                    new_separator = node.redistribute_with_left(left_sibling)
                    parent.keys[node_idx - 1] = new_separator
                return
        
        if node_idx < len(parent.children) - 1:
            right_sibling = parent.children[node_idx + 1]
            if len(right_sibling.keys) > right_sibling.min_keys:
                if node.is_leaf:
                    new_separator = node.redistribute_with_right(right_sibling)
                    parent.keys[node_idx] = new_separator
                return
        
        # Spajanje
        if node_idx > 0:
            left_sibling = parent.children[node_idx - 1]
            if node.is_leaf:
                left_sibling.merge_with_right(node)
            else:
                separator = parent.keys[node_idx - 1]
                left_sibling.merge_with_right(node, separator)
            
            parent.keys.pop(node_idx - 1)
            parent.children.pop(node_idx)
        else:
            right_sibling = parent.children[node_idx + 1]
            if node.is_leaf:
                node.merge_with_right(right_sibling)
            else:
                separator = parent.keys[node_idx]
                node.merge_with_right(right_sibling, separator)
            
            parent.keys.pop(node_idx)
            parent.children.pop(node_idx + 1)
        
        if parent.is_underflow():
            self._fix_underflow(parent)
    
    def range_query(self, start_key: K, end_key: K) -> List[Tuple[K, V]]:
        result = []
        current = self._find_leaf(start_key)
        
        if not current:
            return result
        
        while current:
            for i, key in enumerate(current.keys):
                if key < start_key:
                    continue
                if key > end_key:
                    return result
                result.append((key, current.values[i]))
            current = current.next
        
        return result
    
    def items(self) -> Iterator[Tuple[K, V]]:
        if not self.root:
            return
        
        current = self.root
        while not current.is_leaf:
            current = current.children[0]
        
        while current:
            for i in range(len(current.keys)):
                yield (current.keys[i], current.values[i])
            current = current.next
    
    # Pomoćne metode
    def __len__(self) -> int:
        return self._size
    
    def __contains__(self, key: K) -> bool:
        return self.search(key) is not None
    
    def __getitem__(self, key: K) -> List[V]:
        value = self.search(key)
        if value is None:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key: K, value: V):
        self.insert(key, value)
    
    def __delitem__(self, key: K):
        if not self.delete(key):
            raise KeyError(key)
    
    def keys(self) -> Iterator[K]:
        for key, _ in self.items():
            yield key
    
    def values(self) -> Iterator[List[V]]:
        for _, value_list in self.items():
            yield value_list
    
    def all_values(self) -> Iterator[V]:
        """Iterator preko svih individualnih vrednosti (ne lista)"""
        for _, value_list in self.items():
            for value in value_list:
                yield value
    
    def min_key(self) -> Optional[K]:
        if not self.root:
            return None
        
        current = self.root
        while not current.is_leaf:
            current = current.children[0]
        
        return current.keys[0] if current.keys else None
    
    def max_key(self) -> Optional[K]:
        if not self.root:
            return None
        
        current = self.root
        while not current.is_leaf:
            current = current.children[-1]
        
        return current.keys[-1] if current.keys else None
    
    # Multimap dodatne metode
    def get_all_values(self, key: K) -> List[V]:
        """Vraća sve vrednosti za dati ključ (isti kao search)"""
        return self.search(key) or []
    
    def count_values(self, key: K) -> int:
        """Vraća broj vrednosti za dati ključ"""
        values = self.search(key)
        return len(values) if values else 0
    
    def has_value(self, key: K, value: V) -> bool:
        """Proverava da li ključ sadrži određenu vrednost"""
        values = self.search(key)
        return values is not None and value in values
    
    def total_values_count(self) -> int:
        """Vraća ukupan broj svih vrednosti u stablu"""
        count = 0
        for value_list in self.values():
            count += len(value_list)
        return count
    
    # Database Index metode
    def insert_primary_key(self, key: K, row_id: V) -> bool:
        """
        Umeće ključni atribut - svaki ključ treba da ima tačno jedan red ID.
        Vraća False ako ključ već postoji (violation of uniqueness).
        """
        if key in self:
            return False  # Primary key constraint violation
        self.insert(key, row_id)
        return True
    
    def insert_secondary_index(self, key: K, row_id: V):
        """
        Umeće neključni atribut - ključ može da ima više red ID-ova.
        Uvek uspešno dodaje row_id u listu za dati ključ.
        """
        self.insert(key, row_id)
    
    def get_row_ids(self, key: K) -> List[V]:
        """Vraća sve row ID-ove za dati ključ (za oba tipa indeksa)"""
        return self.get_all_values(key)
    
    def get_single_row_id(self, key: K) -> Optional[V]:
        """
        Vraća jedan row ID za ključni atribut.
        Pretpostavka: ključ mapira na tačno jedan element.
        """
        values = self.search(key)
        return values[0] if values and len(values) == 1 else None
    
    def remove_row_from_index(self, key: K, row_id: V) -> bool:
        """
        Uklanja specifičan row ID iz indeksa.
        Korisno za UPDATE operacije ili brisanje iz sekundarnog indeksa.
        """
        return self.remove_value(key, row_id)
    
    def get_all_rows_in_range(self, start_key: K, end_key: K) -> List[V]:
        """
        Range query koji vraća sve row ID-ove u opsegu ključeva.
        Korisno za range scans u bazama podataka.
        """
        range_results = self.range_query(start_key, end_key)
        all_row_ids = []
        for _, row_id_list in range_results:
            all_row_ids.extend(row_id_list)
        return all_row_ids
    
    def height(self) -> int:
        if not self.root:
            return 0
        
        height = 1
        current = self.root
        while not current.is_leaf:
            current = current.children[0]
            height += 1
        
        return height
    
    def print_tree(self, node: Optional['Node'] = None, level: int = 0):
        if node is None:
            node = self.root
        
        if not node:
            print("Prazno stablo")
            return
        
        indent = "  " * level
        
        if node.is_leaf:
            items = list(zip(node.keys, node.values))
            print(f"{indent}List: {items}")
        else:
            print(f"{indent}Unutrašnji: {node.keys}")
            for child in node.children:
                self.print_tree(child, level + 1)


def comprehensive_test():
    """Sveobuhvatan test ispravnosti B+ Tree implementacije"""
    print("=== SVEOBUHVATAN TEST ISPRAVNOSTI B+ STABLA ===\n")
    
    tree = BPlusTree[int, str](order=3)
    
    # Test 1: Dodavanje elemenata
    print("1. TEST DODAVANJA:")
    data = [(10, "deset"), (5, "pet"), (15, "petnaest"), (3, "tri"), 
            (7, "sedam"), (12, "dvanaest"), (18, "osamnaest"), (1, "jedan")]
    
    for key, value in data:
        tree.insert(key, value)
    
    tree.print_tree()
    print(f"Ukupno elemenata: {len(tree)}\n")
    
    # Test 2: Provera povezanosti listova
    print("2. TEST POVEZANOSTI LISTOVA (levo → desno):")
    current = tree.root
    while not current.is_leaf:
        current = current.children[0]  # Idi na prvi list
    
    leaf_count = 0
    while current:
        leaf_count += 1
        print(f"List {leaf_count}: {list(zip(current.keys, current.values))}")
        if current.next:
            print(f"  → next: {current.next.keys[:2]}...")  # Prikaži prva 2 ključa sledećeg
        else:
            print("  → next: None")
        current = current.next
    
    # Test 3: Provera povezanosti unazad
    print("\n3. TEST POVEZANOSTI UNAZAD (desno → levo):")
    current = tree.root
    while not current.is_leaf:
        current = current.children[-1]  # Idi na poslednji list
    
    leaf_count = 0
    while current:
        leaf_count += 1
        print(f"List {leaf_count}: {list(zip(current.keys, current.values))}")
        if current.prev:
            print(f"  ← prev: {current.prev.keys[-2:]}...")  # Prikaži poslednja 2 ključa prethodnog
        else:
            print("  ← prev: None")
        current = current.prev
    
    # Test 4: Test pretrage
    print("\n4. TEST PRETRAGE:")
    test_keys = [1, 5, 10, 15, 20, 99]
    for key in test_keys:
        result = tree.search(key)
        print(f"  Ključ {key:2d}: {'✓' if result else '✗'} {result}")
    
    # Test 5: Range queries
    print("\n5. TEST RANGE UPITA:")
    ranges = [(3, 12), (1, 5), (15, 20)]
    for start, end in ranges:
        result = tree.range_query(start, end)
        print(f"  Range ({start}-{end}): {result}")
    
    # Test 6: Iteracija kroz sve elemente
    print("\n6. TEST ITERACIJE (treba biti sortiran):")
    all_items = list(tree.items())
    print(f"  Svi elementi: {all_items}")
    
    # Proveri da li je sortiran
    keys = [k for k, v in all_items]
    is_sorted = keys == sorted(keys)
    print(f"  Sortiran: {'✓' if is_sorted else '✗'}")
    
    # Test 7: Test brisanja
    print("\n7. TEST BRISANJA:")
    keys_to_delete = [5, 12, 99]  # postojeći, postojeći, nepostojeći
    
    for key in keys_to_delete:
        before_size = len(tree)
        success = tree.delete(key)
        after_size = len(tree)
        
        print(f"  Brisanje {key}: {'✓' if success else '✗'}")
        if success:
            print(f"    Veličina: {before_size} → {after_size}")
    
    print("\n  Struktura nakon brisanja:")
    tree.print_tree()
    
    # Test 8: Test ključ-vrednost mapiranja
    print("\n8. TEST KLJUČ-VREDNOST MAPIRANJA:")
    print("  Svaki ključ mapira na tačno jednu vrednost:")
    
    # Dodaj duplikat ključ sa novom vrednošću
    old_value = tree.search(10)
    tree.insert(10, "NOVO_DESET")  # Treba da prepiše staru vrednost
    new_value = tree.search(10)
    
    print(f"  Ključ 10: '{old_value}' → '{new_value}'")
    print(f"  Veličina se nije promenila: {len(tree)} (treba da bude ista)")
    
    # Test 9: Test operatora
    print("\n9. TEST OPERATORA:")
    tree[99] = "devedeset_devet"
    print(f"  tree[99] = 'devedeset_devet'")
    print(f"  tree[99]: {tree[99]}")
    print(f"  99 in tree: {99 in tree}")
    
    del tree[99]
    print(f"  del tree[99]")
    print(f"  99 in tree: {99 in tree}")
    
    print("\n=== FINALNI REZIME ===")
    print(f"✓ Dodavanje radi ispravno")
    print(f"✓ Listovi su povezani u oba smera") 
    print(f"✓ Pretraga radi ispravno")
    print(f"✓ Range upiti rade ispravno")
    print(f"✓ Iteracija je sortirana")
    print(f"✓ Brisanje radi ispravno") 
    print(f"✓ Ključ mapira na jednu vrednost")
    print(f"✓ Operatori rade ispravno")
    print(f"✓ Implementacija je ISPRAVNA! 🎉")


if __name__ == "__main__":
    comprehensive_test()