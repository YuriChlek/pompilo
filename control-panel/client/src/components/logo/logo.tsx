import Link from 'next/link';
import Image from 'next/image';
import logo from '@public/logo/pampilo-logo.svg?url';

export const Logo = () => {
    return (
        <Link href="/">
            <Image src={logo} alt="Pampilo logo" width={100} height={30} />
        </Link>
    );
};
